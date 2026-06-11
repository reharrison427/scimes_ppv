# for each scimes cluster, get ppv structures of trunk, branches, and leaves
# keep structures with minax < sqrt(2) x beam_minax, label them unresolved

import numpy as np
from astropy.table import Table
from astrodendro.analysis import PPVStatistic, ScalarStatistic

def compute_ppv(struct, metadata):

    ppv = PPVStatistic(struct, metadata)

    spec_axis = metadata["spectral_axis"]

    v_index = ppv.v_cen.value

    v_cen = np.interp(
        v_index,
        np.arange(len(spec_axis)),
        spec_axis.value
    ) * spec_axis.unit

    return {
        "major_sigma": ppv.major_sigma,
        "minor_sigma": ppv.minor_sigma,
        "sigma_v": ppv.v_rms,
        "flux": ppv.flux,
        "x_cen": ppv.x_cen,
        "y_cen": ppv.y_cen,
        "v_cen": v_cen
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
            "flux_err": np.nan,
            "v_cen_err": np.nan
        }

    results = []

    for _ in range(n_iter):
        resampled = np.random.choice(values, size=len(values), replace=True)
        stat = ScalarStatistic(resampled, indices)
        ppv = PPVStatistic(stat, metadata)
        spec_axis = metadata["spectral_axis"]

        v_index = ppv.v_cen.value

        v_cen = np.interp(
            v_index,
            np.arange(len(spec_axis)),
            spec_axis.value
        ) * spec_axis.unit

        results.append([
            ppv.major_sigma.value,
            ppv.minor_sigma.value,
            ppv.v_rms.value,
            ppv.flux.value,
            v_cen
        ])

    results = np.array(results)

    return {
        "major_sigma_err": np.std(results[:, 0]),
        "minor_sigma_err": np.std(results[:, 1]),
        "sigma_v_err": np.std(results[:, 2]),
        "flux_err": np.std(results[:, 3]),
        "v_cen_err": np.std(results[:, 4])
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
    if min_sigma < beam_sigma:
        return False

    # optional: major axis sanity check
    if enforce_major and (maj_sigma < beam_sigma):
        return False

    return True

def is_unresolved(stats, metadata, enforce_major=True):

    beam_sigma = metadata["beam_major"] / np.sqrt(8 * np.log(2))

    min_sigma = stats.get("minor_sigma_dec", np.nan)
    maj_sigma = stats.get("major_sigma_dec", np.nan)

    if not np.isfinite(min_sigma) or not np.isfinite(maj_sigma):
        return True

    if min_sigma < beam_sigma:
        return True

    if enforce_major and (maj_sigma < beam_sigma):
        return True

    return False

def build_dendro_catalog(dclust, cube, metadata, outfile="catalog.csv", compute_errors=False, n_iter=100, allow_shared_structures=True):
    dendro = dclust.dendrogram

    results = []

    # optional duplicate prevention
    processed = set()

    # SCIMES clusters are returned as dendrogram structure IDs
    cluster_trunks = np.asarray(dclust.clusters, dtype=int)

    for trunk_sid in cluster_trunks:

        trunk = dendro[trunk_sid]

        # all structures within this SCIMES hierarchy
        cluster_structures = trunk.descendants + [trunk]

        for struct in cluster_structures:

            sid = int(struct.idx)

            # prevent duplicates if desired - prevent a structure from showing up in multiple mini-dendrograms

            if (not allow_shared_structures) and (sid in processed):
                continue

            processed.add(sid)

            # structure classification
            '''
            if sid == trunk_sid:
                structure_type = "cluster_trunk"

            elif struct.is_leaf:
                structure_type = "leaf"

            else:
                structure_type = "branch"
            '''
            if struct.is_leaf and struct.parent is None:

                structure_type = "disconnected_leaf"

            elif struct.idx == trunk_sid:

                structure_type = "cluster_trunk"

            elif struct.is_leaf:

                structure_type = "leaf"

            else:

                structure_type = "branch"
            # PPV statistics

            stats = compute_ppv(struct, metadata)

            # deconvolved sizes
            stats, _ = deconvolve_sizes(stats, metadata)

            # label structures where minax < sqrt(2)xbeam minax or majax < beam majax as unresolved

            point_source = is_unresolved(stats, metadata)

            # error calculation

            errors = None

            if compute_errors:

                mask = struct.get_mask()

                values = extract_structure_values(cube, mask)

                indices = np.where(mask)

                errors = bootstrap_errors(
                    values,
                    indices,
                    metadata,
                    n_iter=n_iter
                )

                oversampling_factor = compute_oversampling_scale(
                    metadata,
                    cube
                )

                errors = {
                    k: v * oversampling_factor
                    for k, v in errors.items()
                }

                stats, errors = deconvolve_sizes(
                    stats,
                    metadata,
                    errors
                )

            # ----------------------------------
            # hierarchy information
            # ----------------------------------

            parent_id = (
                int(struct.parent.idx)
                if struct.parent is not None
                else -1
            )

            child_ids = [int(c.idx) for c in struct.children]

            # ----------------------------------
            # catalog row
            # ----------------------------------

            row = {

                # SCIMES cluster assignment
                "cluster_id": int(trunk_sid),

                # dendrogram structure ID
                "structure_id": sid,

                # leaf / branch / cluster_trunk
                "structure_type": structure_type,

                # unresolved structure flag
                "point_source": point_source,

                # hierarchy info
                "parent_id": parent_id,

                "n_children": len(struct.children),

                "child_ids": ",".join(map(str, child_ids))
                if len(child_ids) > 0 else "",

                "level": struct.level,

                "npix": struct.get_npix()
            }

            # add PPV statistics
            row.update(stats)

            # add uncertainties
            if compute_errors:
                row.update(errors)

            results.append(row)

    print(f"Total cataloged structures: {len(results)}")

    table = Table(rows=results)

    # ----------------------------------
    # strip units for CSV output
    # ----------------------------------

    for col in table.colnames:

        if hasattr(table[col], "unit") and table[col].unit is not None:
            table[col] = table[col].value

    table.write(outfile, format="ascii.csv", overwrite=True)

    print(f"Saved catalog to {outfile}")

    return table
    
