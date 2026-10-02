"""Version 2: audited explicit path evidence, not inferred business architecture."""
from pathlib import PurePosixPath
import re
MODULES=['interface','service_api','storage','security','automation']
NAMES={'interface':'Interface / rendering','service_api':'API / service','storage':'Storage / database','security':'Authentication / security','automation':'CI / build / deployment'}
DIRECTORIES={
 'interface':{'frontend','front-end','ui','gui','screens','widgets','styles','css','renderer','webapp'},
 'service_api':{'api','apis','backend','back-end','server','servers','service','services','controllers','controller','routers'},
 'storage':{'db','database','databases','storage','migrations','prisma','sequelize','typeorm','alembic'},
 'security':{'auth','authentication','authorization','security','crypto','cryptography','oauth','oauth2','permissions'},
 'automation':{'ci','cicd','deployment','deployments','docker','terraform','k8s','kubernetes','helm'},
}
FILE_TOKENS={
 'interface':{'ui','gui','renderer'},
 'service_api':{'api','server','controller','router'},
 'storage':{'db','database'},
 'security':{'auth','authentication','authorization','security','crypto','cryptography','oauth','oauth2'},
}
UI_EXT={'.jsx','.tsx','.vue','.svelte','.html','.css','.scss','.sass','.less'}
def classify(path,kind,strict=False):
    p=PurePosixPath(path.lower());dirs=set(p.parts[:-1]);name=p.name
    # CamelCase is split before lowercasing. Generic model/core/lib/src tokens do not qualify.
    stem=PurePosixPath(path).stem
    tokens=set(re.sub(r'([a-z0-9])([A-Z])',r'\1_\2',stem).lower().replace('-','_').replace('.','_').split('_'))
    out={}
    # Extra explicit test conventions absent from the legacy artifact-role classifier.
    explicit_test=any(re.search(r'(^|[._-])(tests?|specs?)([._-]|$)',d) for d in dirs) or bool(re.search(r'(tests?|specs?)\.(cs|java|kt|swift)$',name))
    if kind=='source_candidate' and not explicit_test:
        for category in MODULES[:-1]:
            evidence=sorted(dirs & DIRECTORIES[category])
            if not strict:
                # A marketing HTML file called sales-security-trust is not security implementation evidence.
                matched=tokens & FILE_TOKENS[category]
                if category!='interface' and p.suffix in {'.html','.css','.scss','.sass','.less'}:matched=set()
                evidence+=['file:'+t for t in sorted(matched)]
                if category=='interface' and p.suffix in UI_EXT:evidence+=['extension:'+p.suffix]
                if category=='storage' and p.suffix=='.sql':evidence+=['extension:.sql']
            if evidence:out[category]='|'.join(evidence)
    # Operational artifacts include source/config/otherwise-unclassified files, but not test/doc/vendor/locks.
    if kind in {'source_candidate','configuration_path','other_or_unknown'} and not explicit_test:
        evidence=sorted(dirs & DIRECTORIES['automation'])
        if '.github' in dirs and 'workflows' in dirs:evidence+=['.github/workflows']
        if not strict and (name in {'makefile','gnumakefile','cmakelists.txt','jenkinsfile','.gitlab-ci.yml','.travis.yml','azure-pipelines.yml','build.gradle','build.gradle.kts','pom.xml'} or name.startswith(('dockerfile','docker-compose','compose.y')) or p.suffix=='.tf'):
            evidence+=['build/deploy-file:'+name]
        if evidence:out['automation']='|'.join(evidence)
    return out
