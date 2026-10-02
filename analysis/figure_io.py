"""Matplotlib-only export with fixed physical dimensions and anonymous metadata."""
def save_figure(fig, path, width, height_mm, raster_dpi=220):
    fig.set_size_inches(width / 25.4, height_mm / 25.4)
    metadata = {'Creator': 'Matplotlib', 'Author': None, 'CreationDate': None, 'ModDate': None} if path.suffix == '.pdf' else {'Software': 'Matplotlib'}
    fig.savefig(path, dpi=raster_dpi, metadata=metadata)
