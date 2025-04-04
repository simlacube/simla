# MOTHBALLED FOR STORAGE REASONS
# PLAN TO JUST SUBTRACT THE BG ON THE FLY FOR J1
################################################

import numpy as np
from astropy.io import fits
from photutils import background as bg
from photutils.background import BkgZoomInterpolator as bkginterp
from tqdm import tqdm
# import gzip
# import shutil
# import os

from simla_variables import SimlaVar
from simladb import query, DB_bcdwise
from simla_utils import DN_to_MJypsr

wisepath, bgsub_wisepath = SimlaVar().wisepath, SimlaVar().bgsub_wisepath

def make_wise_bg(image_data):
    # Adapting Cory's code. 
    # returns the same WISE file but with a simple 2D bg subtracted
    bg_2d = bg.Background2D(
        image_data, 
        box_size=(2000,2000), 
        exclude_percentile=90, 
        interpolator=bkginterp(order=1)
        ).background.astype('single')
    return bg_2d

# Get only the WISE files that are touched by IRS observations
unique_irstarget_WISE = np.unique(query(DB_bcdwise.select(DB_bcdwise.WISE_FILE))['WISE_FILE'].to_numpy())

# For each WISE file, subtract a background and convert to MJy/sr
for wfile in tqdm(unique_irstarget_WISE):
    
    wfile = wisepath+wfile
    
    # bgsub_wise = subtract_WISE_background(wfile) # HDU list 
    # conv_bgsub_wise = DN_to_MJypsr(bgsub_wise[0].data, 3)
    # bgsub_wise[0].data = conv_bgsub_wise # MJy/sr
    # outname = (bgsub_wisepath+(wfile.split('/')[-1])).replace('.fits.gz', '_bgsub_mjpsr.fits')
    # bgsub_wise.writeto(outname, overwrite=True)

    image_data = fits.getdata(wfile)
    wise_bg = make_wise_bg(image_data)
    bgsub_data = image_data - wise_bg
    # bgsub_data_conv = DN_to_MJypsr(bgsub_data, 3)
    outname = (bgsub_wisepath+(wfile.split('/')[-1])).replace('.fits.gz', '_bgsub_mjpsr.npy')
    # np.save(outname, bgsub_data_conv)
    np.savez_compressed(outname, data=bgsub_data) 

    # # Compress
    # with open(outname, "rb") as fitsfile:
    #     with gzip.open(outname+'.gz', "wb") as gzfile:
    #         shutil.copyfileobj(fitsfile, gzfile)

    # os.remove(outname)

        