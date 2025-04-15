'''
Create the trimmed masks for IRS BCD suborder, and the trimmed masks for each shard.

"Trimming" refers to the pixels cut from the edges (spatial direction) of suborders. A few pixels
on the edges of the spectral direction are also trimmed for consistent suborder widths.
"Shards" are spatially-separated subdivisions within suborders on the BCD and on sky. They
are indexed from 0-n_shards from left to right on the BCD.

Outputs saved in simlapath/calib.

Make sure edgetrim values and n_shards are set in simla_variables.py
Prerequisite code: none.

'''

import numpy as np
from astropy.io import fits
import os

from simla_variables import SimlaVar

def make_trim_mask(channel_mask, edgetrim):

    '''
    channel_mask: loaded untrimmed fullslit mask array
    edgetrim: % of the width to trim from both sides

    returns to new trimmed mask

    '''
    
    trimmed_mask = []
    for i in range(128):
        row = channel_mask[i]
        onorder_width = int(np.sum(row))
        trim_pix = int(np.floor(onorder_width * (edgetrim/100)))
        trimmed_row = np.zeros(128)
        trimmed_row[np.where(row==1)[0][trim_pix:onorder_width-trim_pix]] = 1
        trimmed_mask.append(trimmed_row)
    trimmed_mask = np.asarray(trimmed_mask)
    return trimmed_mask

def make_shard_mask(channel_mask, n_shards):

    '''
    channel_mask: loaded trimmed fullslit mask array
    n_shards: integer number of shards to divide the suborder into

    returns an array containing the mask for each shard

    '''
    
    shardmasks = [[] for shard in range(n_shards)]
    for i in range(128):
        row = channel_mask[i]
        trimmed_onorder_width = int(np.sum(row))
        shardwidth = trimmed_onorder_width/n_shards
        for shard_id in range(n_shards):
            shard_row = np.zeros(128)
            shard_row[np.where(row==1)[0][int(np.ceil(shardwidth*shard_id)):\
                                          int(np.ceil(shardwidth*(shard_id+1)))]] = 1
            shardmasks[shard_id].append(shard_row)
    shardmasks = np.asarray([np.asarray(shardmask) for shardmask in shardmasks])
    return shardmasks

sl_edgetrim = SimlaVar().sl_edgetrim
ll_edgetrim = SimlaVar().ll_edgetrim
sl_n_shards = SimlaVar().sl_n_shards
ll_n_shards = SimlaVar().ll_n_shards
simlapath = SimlaVar().simlapath

# Load in the original suborder masks
mask_library = [
    [None,
     fits.getdata(simlapath+'calib/masks/irs_mask_SL1.fits'),
     fits.getdata(simlapath+'calib/masks/irs_mask_SL2.fits'),
     fits.getdata(simlapath+'calib/masks/irs_mask_SL3.fits')],
     None,
    [None,
     fits.getdata(simlapath+'calib/masks/irs_mask_LL1.fits'),
     fits.getdata(simlapath+'calib/masks/irs_mask_LL2.fits'),
     fits.getdata(simlapath+'calib/masks/irs_mask_LL3.fits')],
]
# Cut off any rows that don't span the whole width
mask_library[0][1][0] = np.zeros(128)
mask_library[0][1][127] = np.zeros(128)
mask_library[0][2][81] = np.zeros(128)
mask_library[0][3][104] = np.zeros(128)
mask_library[0][3][127] = np.zeros(128)
mask_library[2][1][0] = np.zeros(128)
mask_library[2][1][126] = np.zeros(128)
mask_library[2][1][127] = np.zeros(128)
mask_library[2][2][81] = np.zeros(128)

# Make the fullslit trimmed shard masks
sl1_trim, sl2_trim, sl3_trim, ll1_trim, ll2_trim, ll3_trim = \
    make_trim_mask(mask_library[0][1], sl_edgetrim), \
    make_trim_mask(mask_library[0][2], sl_edgetrim), \
    make_trim_mask(mask_library[0][3], sl_edgetrim), \
    make_trim_mask(mask_library[2][1], ll_edgetrim), \
    make_trim_mask(mask_library[2][2], ll_edgetrim), \
    make_trim_mask(mask_library[2][3], ll_edgetrim)

# Make the masks for each shard
sl1_shardm, sl2_shardm, sl3_shardm, ll1_shardm, ll2_shardm, ll3_shardm = \
    make_shard_mask(sl1_trim, sl_n_shards), \
    make_shard_mask(sl2_trim, sl_n_shards), \
    make_shard_mask(sl3_trim, sl_n_shards), \
    make_shard_mask(ll1_trim, ll_n_shards), \
    make_shard_mask(ll2_trim, ll_n_shards), \
    make_shard_mask(ll3_trim, ll_n_shards)

# Set up directories
if not os.path.exists(simlapath+'calib/trimmed_fullslit_masks/'):
    os.mkdir(simlapath+'calib/trimmed_fullslit_masks/')
if not os.path.exists(simlapath+'calib/shard_masks/'):
    os.mkdir(simlapath+'calib/shard_masks/')

# Save everything
np.save(simlapath+'calib/trimmed_fullslit_masks/SL1.npy', sl1_trim)
np.save(simlapath+'calib/trimmed_fullslit_masks/SL2.npy', sl2_trim)
np.save(simlapath+'calib/trimmed_fullslit_masks/SL3.npy', sl3_trim)
np.save(simlapath+'calib/trimmed_fullslit_masks/LL1.npy', ll1_trim)
np.save(simlapath+'calib/trimmed_fullslit_masks/LL2.npy', ll2_trim)
np.save(simlapath+'calib/trimmed_fullslit_masks/LL3.npy', ll3_trim)

np.save(simlapath+'calib/shard_masks/SL1.npy', sl1_shardm)
np.save(simlapath+'calib/shard_masks/SL2.npy', sl2_shardm)
np.save(simlapath+'calib/shard_masks/SL3.npy', sl3_shardm)
np.save(simlapath+'calib/shard_masks/LL1.npy', ll1_shardm)
np.save(simlapath+'calib/shard_masks/LL2.npy', ll2_shardm)
np.save(simlapath+'calib/shard_masks/LL3.npy', ll3_shardm)