#!/usr/bin/env python3
"""Sequential, resumable repository PR census; no PR sampling, no manuscript writes."""
import argparse
import csv
import concurrent.futures
import collections
import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from analyze_github_delivery import file_kind, quarter

START = '2025-01-01T00:00:00Z'
END = '2026-07-01T00:00:00Z'
QUARTERS = ['2025Q1', '2025Q2', '2025Q3', '2025Q4', '2026Q1', '2026Q2']
AGENTS = ['codex', 'copilot', 'claude_code', 'jules', 'devin']
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/github_repo_contribution_20260924'
RATE = 'rateLimit { cost remaining resetAt }'
FIELDS = '''id number url title body state createdAt updatedAt closedAt mergedAt
 additions deletions changedFiles headRefName baseRefName
 author { __typename login } repository { id nameWithOwner }
 labels(first:100) { totalCount nodes { name } }
 files(first:100) { totalCount pageInfo { hasNextPage endCursor }
 nodes { path additions deletions changeType } }'''
COUNT_QUERY = 'query($q:String!){'+RATE+' search(query:$q,type:ISSUE,first:1){issueCount}}'
PAGE_QUERY = 'query($q:String!,$cursor:String,$pageSize:Int=50,$filePageSize:Int=10){'+RATE+''' search(query:$q,type:ISSUE,first:$pageSize,after:$cursor){
 issueCount pageInfo { hasNextPage endCursor } nodes {... on PullRequest {'''+FIELDS.replace('files(first:100)','files(first:$filePageSize)')+'}}}}'
FILES_QUERY = 'query($id:ID!,$cursor:String){'+RATE+'''node(id:$id){... on PullRequest {
 id files(first:100,after:$cursor){totalCount pageInfo{hasNextPage endCursor}
 nodes{path additions deletions changeType}}}}}'''
FIX = re.compile(r'\b(fix(?:es|ed|ing)?|bugs?|regressions?|hotfix(?:es)?)\b', re.I)
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})')


def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def dt(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00'))


def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(p)


def csvout(p, rows):
    if not rows:
        return
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def read_jsonl(path):
    # str.splitlines() also splits U+0085/U+2028/U+2029 inside valid JSON strings.
    # JSONL records are separated by physical newline bytes, so iterate the file.
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class Deadline(Exception):
    pass


class RequestGate:
    """One shared request budget/cooldown across the bounded network workers."""
    def __init__(self, workers=2, spacing=2.0):
        self.limit = workers; self.spacing = spacing
        self.cv = threading.Condition(); self.active = 0
        self.next_start = 0.0; self.blocked_until = 0.0
        self.remaining = None; self.reset_at = None; self.cancelled = False

    def acquire(self, api):
        while True:
            api.check_time()
            with self.cv:
                if self.cancelled:
                    raise RuntimeError('Concurrent collection cancelled after another worker failed')
                current = time.monotonic()
                wait = max(self.blocked_until, self.next_start) - current
                if self.active < self.limit and wait <= 0:
                    self.active += 1; self.next_start = current + self.spacing
                    return
                self.cv.wait(timeout=min(1.0, max(.05, wait)) if wait > 0 else .2)

    def release(self):
        with self.cv:
            self.active -= 1; self.cv.notify_all()

    def cooldown(self, seconds, secondary=False):
        with self.cv:
            self.blocked_until = max(self.blocked_until, time.monotonic() + seconds)
            if secondary:
                self.limit = 1
            self.cv.notify_all()

    def observe(self, rate):
        reset = dt(rate['resetAt'])
        with self.cv:
            if self.reset_at is None or reset > self.reset_at:
                self.reset_at = reset; self.remaining = rate['remaining']
            elif reset == self.reset_at:
                self.remaining = min(self.remaining, rate['remaining'])
            if self.remaining is not None and self.remaining < 10:
                seconds = max(0, (self.reset_at-datetime.now(timezone.utc)).total_seconds()) + 2
                self.blocked_until = max(self.blocked_until, time.monotonic()+seconds)


class API:
    def __init__(self, deadline=None, gate=None):
        token = os.getenv('GITHUB_TOKEN') or os.getenv('GH_TOKEN')
        if not token:
            token = (ROOT / 'key').read_text().strip()
        if not token.startswith(('ghp_', 'github_pat_', 'gho_')):
            raise ValueError('No recognized GitHub credential; values are never logged')
        if os.getenv('AGENTBENCH_HTTP_TRANSPORT') == 'curl':
            from github_curl_transport import CurlSession
            self.s = CurlSession()
        else:
            self.s = requests.Session()
        self.s.headers.update({'Authorization': 'Bearer ' + token,
                               'Accept': 'application/vnd.github+json',
                               'User-Agent': 'AgentBench-repository-census'})
        self.deadline = dt(deadline) if deadline else None
        self.requests = 0
        self.cost = 0
        self.last_rate = {}
        self.gate = gate

    def wait_seconds(self, seconds):
        end = time.monotonic() + max(0, seconds)
        while time.monotonic() < end:
            self.check_time()
            time.sleep(min(1, end-time.monotonic()))

    def check_time(self):
        if self.deadline and datetime.now(timezone.utc) >= self.deadline:
            raise Deadline('Configured collection deadline reached')

    def request(self, method, url, **kwargs):
        for attempt in range(6):
            self.check_time()
            try:
                if self.gate:
                    self.gate.acquire(self)
                try:
                    r = self.s.request(method, url, timeout=50, **kwargs)
                finally:
                    if self.gate:
                        self.gate.release()
                self.requests += 1
                if r.status_code in (429, 502, 503, 504) or (
                    r.status_code == 403 and ('rate limit' in r.text.lower())):
                    limited = r.status_code in (403,429)
                    if r.headers.get('Retry-After'):
                        wait = float(r.headers['Retry-After'])
                    elif r.headers.get('x-ratelimit-remaining') == '0':
                        wait = max(1, float(r.headers.get('x-ratelimit-reset',time.time()+60))-time.time()+2)
                    else:
                        wait = 60*(2**attempt) if limited else 3*(attempt+1)
                    if self.gate:
                        self.gate.cooldown(wait, secondary=limited)
                    print(json.dumps({'event': 'request_retry', 'status': r.status_code, 'wait_seconds': wait}), flush=True)
                    self.wait_seconds(wait); continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                if attempt == 5:
                    raise RuntimeError('GitHub request failed after retries; credential not logged') from None
                time.sleep(min(20, 2 ** attempt))
        raise RuntimeError('GitHub rate/service unavailable after retries')

    def gql(self, query, variables=None):
        for attempt in range(5):
            if self.last_rate.get('remaining', 100) < 5:
                reset = dt(self.last_rate['resetAt'])
                while datetime.now(timezone.utc) < reset:
                    self.check_time()
                    time.sleep(min(30, max(1, (reset - datetime.now(timezone.utc)).total_seconds())))
            d = self.request('POST', 'https://api.github.com/graphql', json={'query': query, 'variables': variables or {}})
            data = d.get('data') or {}
            if data.get('rateLimit'):
                self.last_rate = data['rateLimit']; self.cost += self.last_rate['cost']
                if self.gate:
                    self.gate.observe(self.last_rate)
            if not d.get('errors'):
                return data
            msgs = [e.get('message', '') for e in d['errors']]
            if any('rate limit' in m.lower() for m in msgs):
                wait = 60*(2**attempt)
                if self.last_rate.get('remaining',1)==0:
                    wait=max(wait,(dt(self.last_rate['resetAt'])-datetime.now(timezone.utc)).total_seconds()+2)
                if self.gate:
                    self.gate.cooldown(wait, secondary=True)
                self.wait_seconds(wait); continue
            if any('timeout' in m.lower() or 'something went wrong' in m.lower() for m in msgs):
                self.wait_seconds(min(30, 3 * (attempt + 1))); continue
            raise RuntimeError('GraphQL: ' + SECRET.sub('[REDACTED]', '; '.join(msgs)))
        raise RuntimeError('GraphQL retries exhausted')


def exclusion(r):
    topics = set(r.get('topics', [])); desc = (r.get('description') or '').lower()
    if r.get('language') in (None, 'Markdown', 'Jupyter Notebook'):
        return 'non_programming_primary_language'
    if topics & {'awesome', 'awesome-list', 'roadmap', 'study-plan', 'books', 'courses',
                 'coding-interview', 'interview-prep', 'interview-practice', 'interview-preparation'}:
        return 'resource_or_study_repository'
    if re.search(r'\b(curated list|collective list|list of|study plan|roadmaps|skills for real engineers|freely available programming books)\b', desc):
        return 'resource_collection_description'
    return ''


def select(api):
    frozen = OUT / 'repositories.json'
    if frozen.exists():
        return json.loads(frozen.read_text())
    source = json.loads((OUT / 'high_star_search_raw.json').read_text())
    if source.get('incomplete_results') or len(source['items']) < 100:
        raise ValueError('Incomplete ranking discovery')
    items = source['items']
    assert all(a['stargazers_count'] >= b['stargazers_count'] for a, b in zip(items, items[1:]))
    eligibility = [dict(repository=r['full_name'], global_search_rank=i+1,
                        stars=r['stargazers_count'], language=r['language'], exclusion=exclusion(r))
                   for i, r in enumerate(items)]
    csvout(OUT / 'ranking_eligibility.csv', eligibility)
    high = [r for r in items if not exclusion(r)][:10]
    result = []
    for i, r in enumerate(high, 1):
        result.append({'repository': r['full_name'], 'repository_id': r['id'], 'group': 'high',
                       'rank': i, 'pair_id': i, 'stars': r['stargazers_count'],
                       'language': r['language'], 'created_at': r['created_at'],
                       'size_kb': r['size'], 'topics': r.get('topics', []),
                       'exists_before_window_end': r['created_at'] < END})
    # Freeze the high list now; low controls are independently discovered after high census.
    dump(frozen, result)
    dump(OUT / 'selection_protocol.json', {
        'ranking_snapshot_file': 'high_star_search_raw.json', 'frozen_at': now(),
        'ranking_query': 'stars:>10000 fork:false archived:false, stars descending, first 100',
        'eligibility': 'Programming primary language; exclude resource/study topics and explicit collection descriptions using exclusion() v1. This is an operational code-project screen, not a universal code-project taxonomy.',
        'code_project_boundary': 'Includes executable skills frameworks and algorithm implementations, and mixed curriculum/software projects. Unvalidated semantic project classification.',
        'high_projects_created_after_observation': [r['repository'] for r in result if not r['exists_before_window_end']],
        'low_control_plan': 'Independent GitHub candidates, 0-10 current stars, same primary language, non-fork/non-archived, pushed since 2025-01-01; deterministic age/size/topic matching without Agent evidence.',
        'period_start': START, 'period_end_exclusive': END,
        'scope': 'All PRs created in window, plus older PRs merged in window; all states for created cohort; no PR subsampling.',
        'main_denominator': 'PRs merged in window; quarters use mergedAt. Created-cohort ratios use createdAt separately.',
        'snapshot_bias': 'Current stars do not represent historical stars; rankings condition on survival. New projects have no exposure before creation.'})
    return result


def low_controls(api, repos):
    if any(r['group'] == 'low' for r in repos):
        return repos
    used = set(); pools = {}
    for h in repos[:10]:
        lang = h['language']
        if lang not in pools:
            p = OUT / ('low_candidates_' + re.sub(r'\W', '_', lang) + '.json')
            if p.exists():
                d = json.loads(p.read_text())
            else:
                d = api.request('GET', 'https://api.github.com/search/repositories', params={
                    'q': f'language:"{lang}" stars:0..10 fork:false archived:false pushed:>=2025-01-01',
                    'sort': 'stars', 'order': 'desc', 'per_page': 100})
                dump(p, d)
            if d.get('incomplete_results'):
                raise ValueError('Low candidate search incomplete')
            pools[lang] = [v for v in d['items'] if not exclusion(v)]
        def score(v):
            age = abs((dt(v['created_at']) - dt(h['created_at'])).days) / 365.25
            size = abs(math.log1p(v['size']) - math.log1p(h['size_kb']))
            a, b = set(v.get('topics', [])), set(h['topics'])
            topical = 1 - len(a & b) / max(1, len(a | b))
            return round(age + size + 2 * topical, 6)
        candidates = sorted((v for v in pools[lang] if v['id'] not in used), key=lambda v: (score(v), v['full_name']))
        if not candidates:
            raise ValueError('No independent low-star candidate for ' + lang)
        v = candidates[0]; used.add(v['id'])
        repos.append({'repository': v['full_name'], 'repository_id': v['id'], 'group': 'low',
                      'rank': h['rank'], 'pair_id': h['pair_id'], 'stars': v['stargazers_count'],
                      'language': v['language'], 'created_at': v['created_at'], 'size_kb': v['size'],
                      'topics': v.get('topics', []), 'exists_before_window_end': v['created_at'] < END,
                      'match_score': score(v), 'matched_to': h['repository']})
    dump(OUT / 'repositories.json', repos)
    return repos


def query_for(repo, kind, a, b):
    datefield = 'created' if kind == 'created' else 'merged'
    stamp = lambda v: v.strftime('%Y-%m-%dT%H:%M:%SZ')
    q = f'repo:{repo} is:pr {datefield}:{stamp(a)}..{stamp(b - timedelta(seconds=1))}'
    if kind == 'older_merged':
        q += ' created:<2025-01-01'
    return q


def connect(folder):
    folder.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(folder / 'state.sqlite3', timeout=30); con.row_factory = sqlite3.Row
    con.execute('PRAGMA busy_timeout=30000')
    con.executescript('''
      CREATE TABLE IF NOT EXISTS windows(q TEXT PRIMARY KEY, parent TEXT, status TEXT,
        expected INTEGER, cursor TEXT, last_error TEXT);
      CREATE TABLE IF NOT EXISTS prs(id TEXT PRIMARY KEY, payload TEXT, collected_at TEXT);
      CREATE TABLE IF NOT EXISTS window_prs(q TEXT, id TEXT, PRIMARY KEY(q,id));
    '''); con.commit()
    return con


def window(api, con, repo, kind, a, b, parent=None):
    q = query_for(repo, kind, a, b)
    row = con.execute('SELECT * FROM windows WHERE q=?', (q,)).fetchone()
    if row and row['status'] == 'complete':
        return
    if row and row['status'] == 'unavailable':
        raise RuntimeError(row['last_error'] or 'Unresolvable PR identities retained')
    if not row:
        count = api.gql(COUNT_QUERY, {'q': q})['search']['issueCount']
        con.execute('INSERT INTO windows VALUES(?,?,?,?,?,?)', (q, parent, 'pending', count, None, None)); con.commit()
        row = con.execute('SELECT * FROM windows WHERE q=?', (q,)).fetchone()
    if row['expected'] > 800 or row['status'] == 'split':
        if (b - a).total_seconds() < 2:
            raise RuntimeError('Search cap cannot be resolved at one-second granularity')
        mid = a + timedelta(seconds=int((b-a).total_seconds()) // 2)
        con.execute('UPDATE windows SET status=? WHERE q=?', ('split', q)); con.commit()
        window(api, con, repo, kind, a, mid, q); window(api, con, repo, kind, mid, b, q)
        return
    cursor = row['cursor']
    if row['status'] == 'mismatch':
        cursor = None
        con.execute('DELETE FROM window_prs WHERE q=?', (q,)); con.commit()
    while True:
        page = fetch_search_page(api,q,cursor)
        if page['issueCount'] > 900:
            con.execute('UPDATE windows SET status=? WHERE q=?', ('split', q)); con.commit()
            return window(api, con, repo, kind, a, b, parent)
        nodes = page['nodes']
        for p in nodes:
            if not p or not p.get('id'):
                raise RuntimeError('Search returned an inaccessible or non-PR node')
            payload = SECRET.sub('[REDACTED_GITHUB_TOKEN]', json.dumps(p, ensure_ascii=False))
            con.execute('INSERT OR REPLACE INTO prs VALUES(?,?,?)', (p['id'], payload, now()))
            con.execute('INSERT OR IGNORE INTO window_prs VALUES(?,?)', (q, p['id']))
        info = page['pageInfo']; next_cursor = info['endCursor']
        if info['hasNextPage'] and (not nodes or not next_cursor or next_cursor == cursor):
            raise RuntimeError('Pagination did not advance')
        n = con.execute('SELECT COUNT(*) FROM window_prs WHERE q=?', (q,)).fetchone()[0]
        status = 'pending' if info['hasNextPage'] else ('complete' if n == page['issueCount'] else 'mismatch')
        con.execute('UPDATE windows SET status=?,expected=?,cursor=? WHERE q=?',
                    (status, page['issueCount'], next_cursor, q)); con.commit()
        total = con.execute('SELECT COUNT(*) FROM prs').fetchone()[0]
        print(json.dumps({'event': 'page', 'at': now(), 'repository': repo, 'records_saved': total,
                          'window_count': n, 'window_expected': page['issueCount'],
                          'rate': api.last_rate}), flush=True)
        if not info['hasNextPage']:
            if status != 'complete':
                raise RuntimeError(f'Search count mismatch: expected {page["issueCount"]}, returned {n}')
            break
        cursor = next_cursor


def fetch_search_page(api,q,cursor):
    for size in [50,25,10,1]:
        try:
            return api.gql(PAGE_QUERY,{'q':q,'cursor':cursor,'pageSize':size,'filePageSize':10})['search']
        except Deadline:
            raise
        except RuntimeError as e:
            transient=any(k in str(e).lower() for k in ['retries','timeout','unavailable','502','503','504'])
            if not transient or size==1:
                raise
            print(json.dumps({'event':'reduce_query_size','at':now(),'failed_page_size':size,
                              'cursor_preserved':True}),flush=True)


def plan_parallel_windows(api, con, repo):
    todo = collections.deque((kind,dt(START),dt(END),None) for kind in ['created','older_merged'])
    leaves = []
    while todo:
        batch = [todo.popleft() for _ in range(min(20,len(todo)))]
        rows = {}; unknown = []
        for i,(kind,a,b,parent) in enumerate(batch):
            q=query_for(repo,kind,a,b)
            row=con.execute('SELECT * FROM windows WHERE q=?',(q,)).fetchone()
            if row:
                rows[i]=dict(row)
            else:
                unknown.append((i,q,parent))
        if unknown:
            query='query{'+RATE
            for i,q,_ in unknown:
                query+=f' w{i}:search(query:{json.dumps(q)},type:ISSUE,first:1)'+'{issueCount}'
            data=api.gql(query+'}')
            for i,q,parent in unknown:
                count=data['w'+str(i)]['issueCount']
                con.execute('INSERT INTO windows VALUES(?,?,?,?,?,?)',(q,parent,'pending',count,None,None))
                rows[i]={'q':q,'parent':parent,'status':'pending','expected':count,'cursor':None}
            con.commit()
        for i,item in enumerate(batch):
            kind,a,b,parent=item;row=rows[i];q=row['q']
            if row['status']=='complete':
                continue
            if row['expected']>800 or row['status']=='split':
                if (b-a).total_seconds()<2:
                    raise RuntimeError('Search cap cannot be resolved at one second')
                mid=a+timedelta(seconds=int((b-a).total_seconds())//2)
                con.execute('UPDATE windows SET status=? WHERE q=?',('split',q))
                todo.extend([(kind,a,mid,q),(kind,mid,b,q)])
            elif row['expected']==0:
                con.execute('UPDATE windows SET status=? WHERE q=?',('complete',q))
            else:
                leaves.append(item)
        con.commit()
    return leaves


def collect_parallel_windows(api, folder, con, repo, workers):
    leaves=plan_parallel_windows(api,con,repo)
    print(json.dumps({'event':'parallel_plan','at':now(),'repository':repo,
                      'pending_windows':len(leaves),'workers':workers}),flush=True)
    def collect(item):
        deadline=api.deadline.isoformat() if api.deadline else None
        client=API(deadline,gate=api.gate);db=connect(folder)
        try:
            kind,a,b,parent=item
            window(client,db,repo,kind,a,b,parent)
            return client.requests,client.cost,client.last_rate
        finally:
            db.close()
    tasks=iter(leaves)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending=set()
        for _ in range(workers):
            item=next(tasks,None)
            if item is not None:pending.add(pool.submit(collect,item))
        try:
            while pending:
                done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    requests,cost,rate=future.result()
                    api.requests+=requests;api.cost+=cost
                    if rate:api.last_rate=rate
                    item=next(tasks,None)
                    if item is not None:pending.add(pool.submit(collect,item))
        except Exception:
            if api.gate:
                with api.gate.cv:
                    api.gate.cancelled=True;api.gate.cv.notify_all()
            for future in pending:future.cancel()
            raise


def agents(p):
    branch = (p.get('headRefName') or '').lower(); body = p.get('body') or ''
    author = ((p.get('author') or {}).get('login') or '').lower()
    hits = {}
    if branch.startswith('codex/'):
        hits['codex'] = 'head:codex/'
    if branch.startswith('copilot/'):
        hits['copilot'] = 'head:copilot/'
    if re.search(r'Co-Authored-By:\s*Claude\b|Generated with Claude Code', body, re.I):
        hits['claude_code'] = 'explicit_claude_body_signature'
    if author in {'google-labs-jules[bot]', 'google-labs-jules'}:
        hits['jules'] = 'jules_author'
    if author in {'devin-ai-integration[bot]', 'devin-ai-integration'}:
        hits['devin'] = 'devin_author'
    return hits


def metrics(p):
    conn = p.get('files') or {}; files = conn.get('nodes') or []
    paths = [f['path'] for f in files]
    complete = (not conn.get('pageInfo', {}).get('hasNextPage', True)
                and len(paths) == len(set(paths)) == conn.get('totalCount') == p['changedFiles'])
    sums_match = (sum(f['additions'] for f in files) == p['additions']
                  and sum(f['deletions'] for f in files) == p['deletions'])
    src = [f for f in files if file_kind(f['path']) == 'source_candidate']
    hit = agents(p)
    labels = [(n['name'] or '').lower() for n in p['labels']['nodes']]
    bug = any(re.sub('[^a-z0-9]', '', v) in {'bug', 'bugfix', 'typebug', 'kindbug', 'typebugfix'} for v in labels)
    return {'id': p['id'], 'repository': p['repository']['nameWithOwner'], 'number': p['number'],
            'url': p['url'], 'created_at': p['createdAt'], 'merged_at': p['mergedAt'],
            'created_quarter': quarter(p['createdAt']), 'merged_quarter': quarter(p['mergedAt']) if p['mergedAt'] else '',
            'created_in_window': START <= p['createdAt'] < END,
            'merged_in_window': bool(p['mergedAt'] and START <= p['mergedAt'] < END),
            'state': p['state'], 'agent_detected': bool(hit), 'agents': '|'.join(sorted(hit)),
            'evidence': json.dumps(hit, ensure_ascii=False), 'fix_title_candidate': bool(FIX.search(p['title'])),
            'bug_label_candidate': bug, 'changed_files': p['changedFiles'],
            'additions': p['additions'], 'deletions': p['deletions'],
            'files_complete': complete, 'file_line_sums_match': sums_match,
            'source_additions': sum(f['additions'] for f in src) if complete and sums_match else None,
            'source_deletions': sum(f['deletions'] for f in src) if complete and sums_match else None}


def export_repo(folder, repo, con, status, error=None):
    raw = [json.loads(r['payload']) for r in con.execute('SELECT payload FROM prs ORDER BY id')]
    ms = [metrics(p) for p in raw]
    with (folder / 'prs.jsonl').open('w') as f:
        for p in raw:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    csvout(folder / 'pr_metrics.csv', ms)
    wins = [dict(r) for r in con.execute('SELECT * FROM windows ORDER BY q')]
    dump(folder / 'search_windows.json', wins)
    leaves = [v for v in wins if v['status'] != 'split']
    expected = sum(v['expected'] for v in leaves)
    valid = bool(wins) and all(v['status'] == 'complete' for v in leaves) and expected == len(ms)
    # Empty after-period repositories still run the two zero-count windows.
    summary = {'repository': repo['repository'], 'group': repo['group'], 'rank': repo['rank'],
               'status': status, 'updated_at': now(), 'unique_prs': len(ms), 'expected_prs': expected,
               'enumeration_complete': valid, 'file_lists_complete': sum(v['files_complete'] for v in ms),
               'source_lines_eligible': sum(v['source_additions'] is not None for v in ms),
               'created_in_window': sum(v['created_in_window'] for v in ms),
               'merged_in_window': sum(v['merged_in_window'] for v in ms),
               'agent_detected': sum(v['agent_detected'] for v in ms), 'error': error}
    summary['merged_file_lists_complete'] = sum(v['merged_in_window'] and v['files_complete'] for v in ms)
    summary['merged_source_lines_eligible'] = sum(v['merged_in_window'] and v['source_additions'] is not None for v in ms)
    summary['file_pagination_scope'] = 'All merged-in-window PRs; other PRs retain first file page and aggregate additions/deletions'
    if status == 'complete' and not valid:
        raise AssertionError('Cannot mark incomplete repository complete')
    dump(folder / 'status.json', summary)
    return summary


def crawl_repo(api, repo, workers=1):
    folder = OUT / 'repositories' / repo['repository'].replace('/', '__')
    if (folder / 'status.json').exists():
        prev = json.loads((folder / 'status.json').read_text())
        if prev['status'] == 'complete':
            return prev
    started = time.monotonic(); con = connect(folder)
    starting_prs = con.execute('SELECT COUNT(*) FROM prs').fetchone()[0]
    timing_path = folder/'timing.json'
    if timing_path.exists():
        first_started_at = json.loads(timing_path.read_text())['first_started_at']
    else:
        first_started_at = con.execute('SELECT MIN(collected_at) FROM prs').fetchone()[0] or now()
        dump(timing_path, {'first_started_at':first_started_at})
    try:
        if workers>1:
            collect_parallel_windows(api,folder,con,repo['repository'],workers)
        else:
            for kind in ['created', 'older_merged']:
                window(api, con, repo['repository'], kind, dt(START), dt(END))
        pending = []
        for row in con.execute('SELECT id,payload FROM prs').fetchall():
            p = json.loads(row['payload']); conn = p.get('files')
            if p['mergedAt'] and START <= p['mergedAt'] < END and conn and conn['pageInfo']['hasNextPage']:
                pending.append(p)
        # Batched file pagination remains within the current repository.
        # Only merged-in-window PRs need full file paths for the main source-code denominator.
        file_batch_size=20
        while pending:
            batch, pending = pending[:file_batch_size], pending[file_batch_size:]
            query = 'query {' + RATE
            for i, p in enumerate(batch):
                cursor = p['files']['pageInfo']['endCursor']
                query += f' p{i}:node(id:{json.dumps(p["id"])})' + '{... on PullRequest {files(first:100,after:' + json.dumps(cursor) + '){totalCount pageInfo{hasNextPage endCursor} nodes{path additions deletions changeType}}}}'
            query += '}'
            try:
                data = api.gql(query)
            except Deadline:
                raise
            except RuntimeError as e:
                if file_batch_size==1 or not any(k in str(e).lower() for k in ['retries','timeout','unavailable','502','503','504']):
                    raise
                pending=batch+pending;file_batch_size=max(1,file_batch_size//2)
                print(json.dumps({'event':'reduce_file_batch_size','at':now(),'batch_size':file_batch_size}),flush=True)
                continue
            for i, p in enumerate(batch):
                node = data.get('p'+str(i)); conn = p['files']; old_cursor = conn['pageInfo']['endCursor']
                if not node or not node.get('files'):
                    p['_fileCollectionError'] = 'PR/files unavailable during pagination'
                else:
                    nxt = node['files']
                    if nxt['totalCount'] != conn['totalCount'] or (nxt['pageInfo']['hasNextPage'] and nxt['pageInfo']['endCursor'] == old_cursor):
                        p['_fileCollectionError'] = 'File counts changed or cursor did not advance'
                    else:
                        conn['nodes'].extend(nxt['nodes']); conn['pageInfo'] = nxt['pageInfo']
                        if conn['pageInfo']['hasNextPage']:
                            pending.append(p)
                        elif not metrics(p)['files_complete']:
                            p['_fileCollectionError'] = 'Final file count/path uniqueness did not match API metadata'
                con.execute('UPDATE prs SET payload=?,collected_at=? WHERE id=?', (json.dumps(p, ensure_ascii=False), now(), p['id']))
            con.commit()
            print(json.dumps({'event':'file_batch','at':now(),'repository':repo['repository'],
                              'remaining_prs':len(pending),'rate':api.last_rate}),flush=True)
        result = export_repo(folder, repo, con, 'complete')
    except Exception as e:
        result = export_repo(folder, repo, con, 'partial', SECRET.sub('[REDACTED]', str(e)))
        if isinstance(e, Deadline):
            result['deadline_reached'] = True
    finally:
        con.close()
    result['elapsed_seconds_this_run'] = round(time.monotonic() - started, 2)
    result['elapsed_seconds_total'] = round((dt(now())-dt(first_started_at)).total_seconds(), 2)
    result['new_prs_this_run'] = max(0,result['unique_prs']-starting_prs)
    result['throughput_new_prs_per_second'] = result['new_prs_this_run']/max(1,result['elapsed_seconds_this_run'])
    dump(folder / 'status.json', result)
    print(json.dumps({'event': 'repository_finished', **result}, ensure_ascii=False), flush=True)
    return result


def summary_rows(ms, repo, valid):
    rows = []
    for cohort, datecol in [('merged', 'merged_quarter'), ('created', 'created_quarter')]:
        for q in QUARTERS + ['ALL']:
            base = [m for m in ms if m[cohort+'_in_window'] and (q == 'ALL' or m[datecol] == q)]
            for task in ['all', 'fix_title_candidate']:
                den = [m for m in base if task == 'all' or m['fix_title_candidate']]
                for agent in ['any'] + AGENTS:
                    num = [m for m in den if m['agent_detected'] and (agent == 'any' or agent in m['agents'].split('|'))]
                    allsrc = all(m['source_additions'] is not None for m in den)
                    source_a = sum(m['source_additions'] or 0 for m in den)
                    source_d = sum(m['source_deletions'] or 0 for m in den)
                    agent_a = sum(m['source_additions'] or 0 for m in num)
                    agent_d = sum(m['source_deletions'] or 0 for m in num)
                    totalch = sum(m['additions']+m['deletions'] for m in den)
                    agentch = sum(m['additions']+m['deletions'] for m in num)
                    pct = lambda a, b, ready=True: 100*a/b if valid and ready and b else None
                    rows.append({'repository': repo['repository'], 'group': repo['group'], 'pair_id': repo['pair_id'],
                                 'stars_snapshot': repo['stars'], 'quarter': q, 'cohort': cohort, 'task': task, 'agent': agent,
                                 'enumeration_complete': valid, 'all_prs': len(den), 'agent_prs': len(num),
                                 'agent_pr_pct': pct(len(num), len(den)), 'all_text_churn': totalch, 'agent_text_churn': agentch,
                                 'agent_text_churn_pct': pct(agentch, totalch), 'source_coverage_complete': allsrc,
                                 'all_source_additions': source_a if allsrc else None,
                                 'agent_source_additions': agent_a if allsrc else None,
                                 'all_source_churn': source_a+source_d if allsrc else None,
                                 'agent_source_churn': agent_a+agent_d if allsrc else None,
                                 'agent_source_additions_pct': pct(agent_a, source_a, allsrc),
                                 'agent_source_churn_pct': pct(agent_a+agent_d, source_a+source_d, allsrc)})
    return rows


def _analyze_unlocked(repos):
    result = []; statuses = []
    for repo in repos:
        folder = OUT / 'repositories' / repo['repository'].replace('/', '__')
        if not (folder / 'status.json').exists():
            statuses.append({'repository': repo['repository'], 'group': repo['group'], 'status': 'pending', 'unique_prs': 0})
            continue
        status = json.loads((folder / 'status.json').read_text()); statuses.append(status)
        ms = [metrics(p) for p in read_jsonl(folder / 'prs.jsonl')]
        result += summary_rows(ms, repo, status['enumeration_complete'])
    csvout(OUT / 'quarter_task_agent_contributions.csv', result)
    dump(OUT / 'progress.json', {'updated_at': now(), 'repositories': statuses,
                                'completed_high': sum(s['group']=='high' and s['status']=='complete' for s in statuses),
                                'completed_low': sum(s['group']=='low' and s['status']=='complete' for s in statuses),
                                'unique_prs_saved': sum(s['unique_prs'] for s in statuses)})
    text = ['# RQ4：高低星仓库贡献占比与修复任务季度趋势（合并原RQ3）', '',
            f'更新：{now()}。观察期：2025-01-01（含）至2026-07-01（不含），UTC。', '',
            '逐仓库完整枚举观察期创建的全部PR，并补入此前创建、观察期合并的PR；不抽样。主贡献指标使用合并日期分季度，创建期活动另表保留。', '',
            '仓库名单与研究范围见 repositories.json、selection_protocol.json；技术项目批次由用户确认，并保存核心源码、构建和维护结构证据。不能称为已穷尽并验证全GitHub技术项目分类的排名。', '',
            '| 仓库 | 组别 | 状态 | 已保存PR |', '|---|---|---|---:|']
    text += [f"| {s['repository']} | {s['group']} | {s['status']} | {s['unique_prs']:,} |" for s in statuses]
    text += ['', '## 完整枚举仓库的主结果', '', '| 仓库 | 同期合并PR | 可识别Agent PR | PR占比 | 全文本变更占比 | 源码候选变更占比 |', '|---|---:|---:|---:|---:|---:|']
    fmt = lambda v: 'NA' if v is None else f'{v:.2f}%'
    for r in result:
        if r['enumeration_complete'] and r['cohort']=='merged' and r['quarter']=='ALL' and r['task']=='all' and r['agent']=='any':
            text.append(f"| {r['repository']} | {r['all_prs']:,} | {r['agent_prs']:,} | {fmt(r['agent_pr_pct'])} | {fmt(r['agent_text_churn_pct'])} | {fmt(r['agent_source_churn_pct'])} |")
    text += ['', '## 原RQ3如何并入RQ4', '',
             '在同一仓库、同一季度内，以全部标题修复候选PR为分母，以其中有可观测Agent证据的PR为分子，同时计算PR数量、新增源码行和源码增删行占比。对应明细表 task=fix_title_candidate、cohort=merged；agent=any 为去重总体，各Agent行可重叠。', '',
             '标题fix/bug/regression/hotfix词形仅定义规则候选，不是验证后的修复任务标签。Agent规则沿用原五类可观测证据：Codex/Copilot分支前缀、Claude正文明确签名、Jules/Devin作者账号；没有识别信号不等于纯人工。未使用评论中的评审Agent作为编码归属。', '',
             '一个PR即使有人类修改，也整体归入“Agent参与PR”修改量，不能解读为Agent独立写出的代码。源码候选采用既有路径规则，排除测试、文档、配置、锁文件和生成/第三方路径；全文本修改量同时提供。不是现存代码存量占比，也不覆盖绕过PR的直接提交。', '',
             '未完整枚举的仓库不发布占比。全文件分页优先覆盖期内已合并PR；其他状态保留首批文件和完整新增/删除总量。源码文件列表或增删行对账不完整时，源码占比为NA；零分母为NA。complete表示PR枚举完成，文件覆盖在status.json另列。当前排名中新建项目在早期季度可能尚不存在；Linux等项目的GitHub PR不代表其全部开发流程。低星对照是有限候选框内的近邻匹配，不能支持星数的因果效应。', '',
             '复现：python3 scripts/collect_repo_contributions.py --mode analyze。原始PR和SQLite检查点位于 repositories/，无需重爬即可开始分析。']
    (OUT / 'RQ4_高低星与季度修复分析.md').write_text('\n'.join(text)+'\n')


def analyze(repos):
    with (OUT/'analysis.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _analyze_unlocked(repos)


def main():
    global OUT
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['run', 'analyze', 'select'], default='run')
    p.add_argument('--deadline', help='Optional UTC deadline; omitted means no time limit'); p.add_argument('--only-repo')
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--workers', type=int, choices=[1,2,3,4,5], default=1)
    p.add_argument('--request-spacing',type=float,default=2.0)
    p.add_argument('--try-next-repo',action='store_true',help='Try the next unfinished repository despite the forecast; retain the hard deadline')
    p.add_argument('--recover-current', action='store_true', help='Only resume the explicitly named existing partial repository until the deadline')
    p.add_argument('--include-low-controls', action='store_true', help='Only use after the low-control selection protocol is approved')
    args = p.parse_args(); OUT = args.output.resolve(); OUT.mkdir(parents=True, exist_ok=True)
    if args.recover_current and not args.only_repo:
        raise SystemExit('--recover-current requires --only-repo')
    if args.mode == 'analyze':
        return analyze(json.loads((OUT / 'repositories.json').read_text()))
    lock = (OUT/'crawler.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another census process holds the lock; not starting a duplicate')
    if args.request_spacing<1.0:
        raise SystemExit('Request spacing must be at least 1 second')
    gate=RequestGate(workers=args.workers,spacing=args.request_spacing) if args.workers>1 else None
    api = API(args.deadline,gate=gate); repos = select(api)
    if args.mode == 'select':
        print(json.dumps(repos, ensure_ascii=False)); return
    dump(OUT / 'run_state.json', {'status': 'running', 'started_at': now(), 'pid': os.getpid(), 'deadline': args.deadline,
                                  'workers':args.workers,'minimum_request_spacing_seconds':gate.spacing if gate else None})
    throughputs = []; stopped = None; trial_available = args.try_next_repo; deferred = []
    try:
        # High-star projects take precedence; finish each repository before starting the next.
        index = 0
        while index < len(repos):
            repo = repos[index]; index += 1
            if args.only_repo and repo['repository'] != args.only_repo:
                continue
            api.check_time()
            folder = OUT / 'repositories' / repo['repository'].replace('/', '__')
            previous = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else {}
            if previous.get('status')=='complete':
                continue
            if previous.get('status') != 'complete':
                counts = {kind: api.gql(COUNT_QUERY, {'q': query_for(repo['repository'], kind, dt(START), dt(END))})['search']['issueCount'] for kind in ['created', 'older_merged']}
                estimated_n = sum(counts.values())
                rate = min(throughputs) if throughputs else 12.0
                existing_n=0
                if (folder/'state.sqlite3').exists():
                    with sqlite3.connect(folder/'state.sqlite3') as existing_db:
                        existing_n=existing_db.execute('SELECT COUNT(*) FROM prs').fetchone()[0]
                remaining_n=max(0,estimated_n-existing_n)
                seconds = 120 + remaining_n / max(.1, rate) * 1.5
                remaining = (api.deadline - datetime.now(timezone.utc)).total_seconds() if api.deadline else None
                estimate = {'repository': repo['repository'], 'expected_prs': estimated_n, 'cohort_counts': counts,
                            'already_saved_prs':existing_n,'remaining_prs_estimate':remaining_n,
                            'estimated_seconds': round(seconds), 'deadline_remaining_seconds': round(remaining) if remaining is not None else None,
                            'measured_prs_per_second': throughputs, 'updated_at': now()}
                dump(OUT / 'current_estimate.json', estimate)
                print(json.dumps({'event': 'repository_start', **estimate}), flush=True)
                recovering = args.recover_current and previous.get('unique_prs',0)>0
                if remaining is not None and seconds + 180 > remaining and not recovering and not trial_available:
                    stopped = 'Stopped before ' + repo['repository'] + ': forecast would exceed deadline'; break
                if trial_available:
                    print(json.dumps({'event':'concurrency_trial','repository':repo['repository'],'workers':args.workers,'deadline':args.deadline}),flush=True)
                    trial_available = False
            result = crawl_repo(api, repo,workers=args.workers); analyze(repos)
            if result['status'] != 'complete':
                if str(result.get('error','')).startswith('Unresolvable PR identities retained'):
                    deferred.append(repo['repository'])
                    # The repository thread pool has joined; preserve shared rate
                    # limits while allowing the next repository to make requests.
                    if api.gate:
                        with api.gate.cv:
                            api.gate.cancelled = False
                            api.gate.cv.notify_all()
                    print(json.dumps({'event':'repository_deferred','repository':repo['repository'],
                                      'reason':result['error'],'at':now()}),flush=True)
                    continue
                stopped = 'Partial repository ' + repo['repository'] + ': ' + str(result.get('error')); break
            speed=result.get('throughput_new_prs_per_second',0)
            if speed>0 and result.get('new_prs_this_run',0)>=100:
                throughputs.append(speed)
            if index == len(repos) and len(repos) == 10 and not args.only_repo and args.include_low_controls:
                repos = low_controls(api, repos); analyze(repos)
    except Exception as e:
        stopped = SECRET.sub('[REDACTED]', str(e))
    if deferred:
        stopped = (stopped+'; ' if stopped else '')+'Unresolvable PR identities retained in: '+', '.join(deferred)
    analyze(repos)
    dump(OUT / 'run_state.json', {'status': 'stopped' if stopped else 'finished', 'stopped_reason': stopped,
                                  'finished_at': now(), 'pid': os.getpid(), 'deadline': args.deadline,
                                  'requests': api.requests, 'graphql_cost': api.cost, 'last_rate': api.last_rate})
    print(json.dumps({'event': 'run_finished', 'reason': stopped, 'at': now()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
