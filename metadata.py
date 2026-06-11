import astropy.units as u
from astropy.constants import c

def build_metadata(cube):

    header = cube.header

    # --- spatial scale ---
    spatial_scale = abs(header['CDELT2']) * u.deg

    # --- beam ---
    beam_major = header.get('BMAJ', 0) * u.deg
    beam_minor = header.get('BMIN', 0) * u.deg

    # --- spectral axis ---
    spec_axis = cube.spectral_axis
    spec_unit = spec_axis.unit

    # ------------------------------------------------------------
    # DETERMINE TYPE OF SPECTRAL AXIS
    # ------------------------------------------------------------

    if spec_unit.is_equivalent(u.m / u.s):
        # already velocity
        velocity_axis = spec_axis

    elif spec_unit.is_equivalent(u.Hz):
        # frequency to velocity
        restfreq = header.get('RESTFRQ', None)
        if restfreq is None:
            raise ValueError("RESTFRQ required to convert frequency to velocity")

        velocity_axis = spec_axis.to(
            u.km/u.s,
            equivalencies=u.doppler_radio(restfreq * u.Hz)
        )

    elif spec_unit.is_equivalent(u.m):
        # wavelength to velocity
        restfreq = header.get('RESTFRQ', None)
        if restfreq is None:
            raise ValueError("RESTFRQ required to convert wavelength to velocity")

        velocity_axis = spec_axis.to(
            u.km/u.s,
            equivalencies=u.doppler_radio(restfreq * u.Hz)
        )

    else:
        raise ValueError(f"Unsupported spectral axis unit: {spec_unit}")

    # ------------------------------------------------------------
    # VELOCITY SCALE
    # ------------------------------------------------------------

    velocity_scale = abs(velocity_axis[1] - velocity_axis[0])

    # ------------------------------------------------------------
    # WAVELENGTH (from rest frequency if available)
    # ------------------------------------------------------------

    restfreq = header.get('RESTFRQ', None)

    if restfreq is not None:
        wavelength = (c / (restfreq * u.Hz)).to(u.m)
    else:
        wavelength = None

    # ------------------------------------------------------------
    # BUILD METADATA
    # ------------------------------------------------------------
    
    metadata = {
        'spatial_scale': spatial_scale,
        'velocity_scale': velocity_scale,
        'beam_major': beam_major,
        'beam_minor': beam_minor,
        'data_unit': cube.unit,
        'spectral_axis': velocity_axis
    }

    if wavelength is not None:
        metadata['wavelength'] = wavelength

    return metadata