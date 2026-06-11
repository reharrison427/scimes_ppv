
import numpy as np
from astropy.table import Table
from astrodendro.analysis import PPVStatistic, ScalarStatistic

def compute_ppv(struct, metadata):
    ppv = PPVStatistic(struct, metadata)
    return {
        "major_sigma": ppv.major_sigma,
        "minor_sigma": ppv.minor_sigma,
        "sigma_v": ppv.v_rms,
        "flux": ppv.flux,
        "x_cen": ppv.x_cen,
        "y_cen": ppv.y_cen,
        "v_cen": ppv.v_cen
    }

# extract values from leaf and cluster structures - necessary for large cubes
def extract_structure_values(cube, mask):

    values = []

    for i in range(cube.shape[0]):
        plane = cube.unmasked_data[i]
        m = mask[i]

        if np.any(m):
            values.append(plane[m])

    if len(values) == 0:
        return np.array([])

    return np.concatenate(values)

# scale errors by number of correlated pixels
# assume all pixels within one beam and in adjacent channels are correlated
# multiply errors by sqrt(2 x pixels per beam)
def compute_oversampling_scale(metadata, cube):

    bmaj = metadata["beam_major"]
    bmin = metadata["beam_minor"]

    # beam area (FWHM)
    beam_area = (np.pi / (4 * np.log(2))) * bmaj * bmin

    # pixel area
    cdelt1 = abs(cube.header["CDELT1"])
    cdelt2 = abs(cube.header["CDELT2"])
    pix_area = cdelt1 * cdelt2 * bmaj.unit**2

    npix_beam = (beam_area / pix_area).decompose().value

    return np.sqrt(2 * npix_beam)

# bootstrap errors
def bootstrap_errors(values, indices, metadata, n_iter=100):

    if len(values) == 0:
        return {
            "major_sigma_err": np.nan,
            "minor_sigma_err": np.nan,
            "sigma_v_err": np.nan,
            "flux_err": np.nan
        }

    results = []

    for _ in range(n_iter):
        resampled = np.random.choice(values, size=len(values), replace=True)
        stat = ScalarStatistic(resampled, indices)
        ppv = PPVStatistic(stat, metadata)

        results.append([
            ppv.major_sigma.value,
            ppv.minor_sigma.value,
            ppv.v_rms.value,
            ppv.flux.value
        ])

    results = np.array(results)

    return {
        "major_sigma_err": np.std(results[:, 0]),
        "minor_sigma_err": np.std(results[:, 1]),
        "sigma_v_err": np.std(results[:, 2]),
        "flux_err": np.std(results[:, 3])
    }


# deconvolve sizes of structures from beam
def deconvolve_sizes(stats, metadata, errors=None):

    beam_maj = metadata["beam_major"]
    beam_min = metadata["beam_minor"]

    beam_sigma_maj = beam_maj / np.sqrt(8 * np.log(2))
    beam_sigma_min = beam_min / np.sqrt(8 * np.log(2))

    maj = stats["major_sigma"]
    min_ = stats["minor_sigma"]

    # --- deconvolution ---
    maj_dec = np.sqrt(np.maximum(maj**2 - beam_sigma_maj**2, 0))
    min_dec = np.sqrt(np.maximum(min_**2 - beam_sigma_min**2, 0))

    stats["major_sigma_dec"] = maj_dec
    stats["minor_sigma_dec"] = min_dec

    radius = 1.91 * np.sqrt(maj_dec * min_dec)
    stats["radius"] = radius


    # --- propagate errors ---
    if errors is not None:

        maj_err = errors.get("major_sigma_err", np.nan)
        min_err = errors.get("minor_sigma_err", np.nan)

        with np.errstate(divide='ignore', invalid='ignore'):
            maj_dec_err = (maj / maj_dec) * maj_err
            min_dec_err = (min_ / min_dec) * min_err

        errors["major_sigma_dec_err"] = maj_dec_err
        errors["minor_sigma_dec_err"] = min_dec_err

        # radius uncertainty propagation

        radius_err = radius * 0.5 * np.sqrt(
            (maj_dec_err / maj_dec)**2 +
            (min_dec_err / min_dec)**2
        )

        errors["radius_err"] = radius_err

    return stats, errors

# cut on structure size: minor axis > sqrt(2)xbeam width
def passes_resolution_cut(stats, metadata, enforce_major=True):
    """
    Check whether a structure passes the resolution criterion.

    Parameters
    ----------
    stats : dict
        Output of compute_ppv + deconvolution step.
        Must contain 'major_sigma_dec' and 'minor_sigma_dec'.
    metadata : dict
        Must contain 'beam_major' (FWHM).
    enforce_major : bool, optional
        If True, also require major axis > beam sigma.

    Returns
    -------
    bool
        True if structure passes the cut, False otherwise.
    """

    # beam FWHM → sigma
    beam_sigma = metadata["beam_major"] / np.sqrt(8 * np.log(2))

    min_sigma = stats.get("minor_sigma_dec", np.nan)
    maj_sigma = stats.get("major_sigma_dec", np.nan)

    # reject invalid values
    if not np.isfinite(min_sigma) or not np.isfinite(maj_sigma):
        return False

    # main cut: minor axis must be resolved
    if min_sigma < np.sqrt(2) * beam_sigma:
        return False

    # optional: major axis sanity check
    if enforce_major and (maj_sigma < beam_sigma):
        return False

    return True

def build_catalog(dclust, cube, metadata, outfile="catalog.csv", compute_errors=False, n_iter=100):

    dendro = dclust.dendrogram
    leaf_labels = dclust.leaves_asgn.data
    cluster_labels = dclust.clusters_asgn.data

    results = []

    # isolated leaves: identify any leaf whose ID does not belong to a cluster ID
    leaf_ids = np.unique(leaf_labels)
    leaf_ids = leaf_ids[leaf_ids > 0]

    cluster_ids = set(np.unique(cluster_labels))
    cluster_ids.discard(0)

    isolated = [lid for lid in leaf_ids if lid not in cluster_ids]

    for lid in isolated:

        struct = dendro[int(lid)]
        stats = compute_ppv(struct, metadata)

        stats, _ = deconvolve_sizes(stats, metadata)

        # resolution cut
        if not passes_resolution_cut(stats, metadata):
            continue

        errors = None

        # --- optional bootstrap ---
        if compute_errors:
            mask = struct.get_mask()
            values = extract_structure_values(cube, mask)
            indices = np.where(mask)
            errors = bootstrap_errors(values, indices, metadata, n_iter=n_iter)
            oversampling_factor = compute_oversampling_scale(metadata, cube)
            errors = {k: v * oversampling_factor for k, v in errors.items()}
            #errors = errors * oversampling_factor

            # --- deconvolution ---
            stats, errors = deconvolve_sizes(stats, metadata, errors)

        row = {"id": int(lid), "type": "leaf"}
        row.update(stats)

        if compute_errors:
            row.update(errors)

        results.append(row)

    n_pass = len(results)

    print(f"Isolated leaves (before cut): {len(isolated)}")
    print(f"Isolated leaves (after cut): {n_pass}")

    # cluster trunks: identify the lowest common ancestor of clusters identified by SCIMES
    for i, struct_ids in enumerate(dclust.clusters):

        struct_ids = np.atleast_1d(struct_ids).astype(int)

        structs = [dendro[j] for j in struct_ids]
        trunk = structs[0]
        for s in structs[1:]:
            trunk = trunk.get_common_ancestor(s)

        stats = compute_ppv(trunk, metadata)

        stats, _ = deconvolve_sizes(stats, metadata)

        # resolution cut
        if not passes_resolution_cut(stats, metadata):
            continue

        errors = None

        if compute_errors:
            mask = trunk.get_mask()
            values = extract_structure_values(cube, mask)
            indices = np.where(mask)
            errors = bootstrap_errors(values, indices, metadata, n_iter=n_iter)
            oversampling_factor = compute_oversampling_scale(metadata, cube)
            errors = {k: v * oversampling_factor for k, v in errors.items()}
            #errors = errors * oversampling_factor
            stats, errors = deconvolve_sizes(stats, metadata, errors)

        row = {"id": int(i), "type": "cluster"}
        row.update(stats)

        if compute_errors:
            row.update(errors)

        results.append(row)

    print(f"Total structures after cut: {len(results)}")

    table = Table(rows=results)

    # strip units for CSV
    for col in table.colnames:
        if hasattr(table[col], "unit") and table[col].unit is not None:
            table[col] = table[col].value

    table.write(outfile, format="ascii.csv", overwrite=True)
    print(f"Saved catalog to {outfile}")

    return table