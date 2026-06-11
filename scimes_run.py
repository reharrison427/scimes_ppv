from astrodendro.analysis import ppv_catalog
from scimes import SpectralCloudstering

def run_scimes(dendro, cube, metadata):
    cat = ppv_catalog(dendro, metadata)

    dclust = SpectralCloudstering(
        dendro,
        cat,
        cube.header,
        save_isol_leaves=True
    )

    return dclust