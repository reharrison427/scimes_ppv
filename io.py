from spectral_cube import SpectralCube
import astropy.units as u

def load_cube(filename):
    cube = SpectralCube.read(filename)
    # ensure velocity axis
    cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio')
    return cube