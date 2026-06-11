import matplotlib.pyplot as plt
import itertools

def plot_spectrum(cube, struct, struct_id, struct_type,
                  outdir="spectra", show=False):

    import os
    os.makedirs(outdir, exist_ok=True)

    mask = struct.get_mask()
    subcube = cube.with_mask(mask)

    spectrum = subcube.mean(axis=(1, 2))
    velocity = cube.spectral_axis

    fig, ax = plt.subplots()

    ax.plot(velocity.value, spectrum.value)

    ax.set_xlabel(f"Velocity [{velocity.unit}]")
    ax.set_ylabel(f"Intensity [{cube.unit}]")

    ax.set_title(f"{struct_type} {struct_id}")

    filename = f"{outdir}/{struct_type}_{struct_id}.png"
    fig.savefig(filename, dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_footprint(cube, struct, struct_id, struct_type,
                   outdir="footprints", show=False):

    import os
    os.makedirs(outdir, exist_ok=True)

    mom0 = cube.moment0()

    mask = struct.get_mask().any(axis=0)

    fig, ax = plt.subplots()

    ax.imshow(mom0.value, origin='lower', cmap='gray')

    ax.contour(mask, levels=[0.5], colors='red', linewidths=1)

    ax.set_title(f"{struct_type} {struct_id}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    filename = f"{outdir}/{struct_type}_{struct_id}.png"
    fig.savefig(filename, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


import numpy as np

def plot_all_structures(dclust, cube, catalog):

    dendro = dclust.dendrogram

    # build lookup of structures that passed cuts
    valid = set(zip(catalog["type"], catalog["id"]))

    # ------------------------------------------------------------
    # ISOLATED LEAVES
    # ------------------------------------------------------------

    leaf_labels = dclust.leaves_asgn.data
    cluster_labels = dclust.clusters_asgn.data

    leaf_ids = np.unique(leaf_labels)
    leaf_ids = leaf_ids[leaf_ids > 0]

    cluster_ids = set(np.unique(cluster_labels))
    cluster_ids.discard(0)

    isolated_leaf_ids = [lid for lid in leaf_ids if lid not in cluster_ids]

    for lid in isolated_leaf_ids:

        if ("leaf", int(lid)) not in valid:
            continue

        struct = dendro[int(lid)]

        plot_spectrum(cube, struct, lid, "leaf")
        plot_footprint(cube, struct, lid, "leaf")

    # ------------------------------------------------------------
    # CLUSTERS (TRUNKS)
    # ------------------------------------------------------------

    def get_lca(dendro, ids):
        ids = np.atleast_1d(ids).astype(int)

        structs = [dendro[i] for i in ids]
        trunk = structs[0]

        for s in structs[1:]:
            trunk = trunk.get_common_ancestor(s)

        return trunk

    for i, struct_ids in enumerate(dclust.clusters):

        if ("cluster", int(i)) not in valid:
            continue

        trunk = get_lca(dendro, struct_ids)

        plot_spectrum(cube, trunk, i, "cluster")
        plot_footprint(cube, trunk, i, "cluster")