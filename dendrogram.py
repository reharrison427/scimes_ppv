from astrodendro import Dendrogram
import numpy as np
from astropy.io import fits
def make_dendrogram(cube, min_value, min_delta, min_npix, pb_image=None, pb_thresh=0.5):
    # if you provide a pb image, only calculate dendrogram where pb > pb_thresh
    if pb_image == None:
        data = cube.unmasked_data[:].value
        
    else:
        data = cube.unmasked_data[:].value
        hdu_pb = fits.open(pb_image)[0]
        data_pb = hdu_pb.data
        
        # take data where pb > 0.5
        data = np.where(data_pb > pb_thresh, data, 0)
    
    dendro = Dendrogram.compute(
            data,
            min_value=min_value,
            min_delta=min_delta,
            min_npix=min_npix
        )
    return dendro