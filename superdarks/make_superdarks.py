'''
For each channel (SL or LL), create "superdarks" for each RAMPTIME and binned by zodi level.

"Superdarks" are zodi-subtracted, sigma-clipped, and median-combined stacks of IRS BCDs where all shards
pass a Judge1 cut.

The Judge1 cut may be different than what is used for cube backgrounds, and is specified in simla_variables.py.
Set sigma_clip and ISM cuts in simla_variables.py.

The signal that the superdarks capture varies with zodi, so superdarks are binned into a number of bins 
specified in simla_variables.py.

Each RAMPTIME has different pixel behaviors, so we separate by that as well.

Prerequisite code: bcd_metadata.py, foreground_model.py, judge1.py (or ..._multip) and prereqs therein.

'''

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from tqdm import tqdm
import os
import gc
from collections import Counter
import pickle

from simladb import query, setup_superdark, DB_bcd, DB_foreground, DB_judge1
from simla_variables import SimlaVar

simlapath = SimlaVar().simlapath
med_pixval_cut = SimlaVar().sd_median_pixval_cut
sl_n_shards, ll_n_shards = SimlaVar().sl_n_shards, SimlaVar().ll_n_shards

# Setup directories
sd_dir = simlapath+'/superdarks/superdarks/'
if not os.path.exists(sd_dir):
    os.mkdir(sd_dir)

zodi_im_path = simlapath+'zodi_images/zodi_images/'

# Load in slit masks for a judge2-like test
shardmask_lib = [
    [np.where(np.load(simlapath+'calib/shard_masks/SL1.npy')==0, np.nan, 1), 
     np.where(np.load(simlapath+'calib/shard_masks/SL2.npy')==0, np.nan, 1),
     np.where(np.load(simlapath+'calib/shard_masks/SL3.npy')==0, np.nan, 1)],
    None,
    [np.where(np.load(simlapath+'calib/shard_masks/LL1.npy')==0, np.nan, 1), 
     np.where(np.load(simlapath+'calib/shard_masks/LL2.npy')==0, np.nan, 1),
     np.where(np.load(simlapath+'calib/shard_masks/LL3.npy')==0, np.nan, 1)],
    None
]

# Collect Judge1-selected dark BCDs
# Not shard-based, all shards on a BCD must qualify
q = query(setup_superdark.select(DB_bcd.DCEID, DB_bcd.FILE_NAME, DB_bcd.MJD_OBS, DB_bcd.RAMPTIME, \
                                 DB_foreground.ZODI_12, \
                                 DB_judge1.CHNLNUM, DB_judge1.SUBORDER, DB_judge1.SHARD) \
                         .where((DB_judge1.BACKSUB_PHOT>=SimlaVar().judge1_sd_cut[0]) \
                               &(DB_judge1.BACKSUB_PHOT<=SimlaVar().judge1_sd_cut[1]) \
                               &(DB_judge1.BACKSUB_PHOT!=0.0) \
                               &(DB_bcd.OBJTYPE.notin(SimlaVar().banned_objtypes)) \
                               &(DB_bcd.OBJECT.notin(SimlaVar().banned_objects)) \
                               &(DB_foreground.ISM_12<=SimlaVar().ism_sd_cut) \
                               &((DB_bcd.CHNLNUM==0)|(DB_bcd.CHNLNUM==2))))
dceids, fnames, mjds, ramps, zodis, chnls, suborders, shardids = \
    q['DCEID'].to_numpy(), q['FILE_NAME'].to_numpy(), q['MJD_OBS'].to_numpy(), \
    q['RAMPTIME'].to_numpy(), q['ZODI_12'].to_numpy(), q['CHNLNUM'].to_numpy(), \
    q['SUBORDER'].to_numpy(), q['SHARD'].to_numpy()

dceid_count = Counter(dceids)
occurence = np.asarray([dceid_count[d] for d in dceids])

# Fulldark DCEIDs have an occurence = shard_num*2
dark_sl_mask = np.where((chnls==0)&(occurence==(2*SimlaVar().sl_n_shards)))
dark_ll_mask = np.where((chnls==2)&(occurence==(2*SimlaVar().ll_n_shards))&(mjds<SimlaVar().ll_gain_change_mjd))
dark_lla_mask = np.where((chnls==2)&(occurence==(2*SimlaVar().ll_n_shards))&(mjds>=SimlaVar().ll_gain_change_mjd))

sl_dark_fnames, sl_dark_ramps, sl_dark_zodis = \
    SimlaVar().irspath+fnames[dark_sl_mask], ramps[dark_sl_mask], zodis[dark_sl_mask]
ll_dark_fnames, ll_dark_ramps, ll_dark_zodis = \
    SimlaVar().irspath+fnames[dark_ll_mask], ramps[dark_ll_mask], zodis[dark_ll_mask]
lla_dark_fnames, lla_dark_ramps, lla_dark_zodis = \
    SimlaVar().irspath+fnames[dark_lla_mask], ramps[dark_lla_mask], zodis[dark_lla_mask]

# From this point on shard info is not needed
_, sl_unique = np.unique(sl_dark_fnames, return_index=True)
sl_dark_fnames, sl_dark_ramps, sl_dark_zodis = \
    sl_dark_fnames[sl_unique], sl_dark_ramps[sl_unique], sl_dark_zodis[sl_unique]
_, ll_unique = np.unique(ll_dark_fnames, return_index=True)
ll_dark_fnames, ll_dark_ramps, ll_dark_zodis = \
    ll_dark_fnames[ll_unique], ll_dark_ramps[ll_unique], ll_dark_zodis[ll_unique]
_, lla_unique = np.unique(lla_dark_fnames, return_index=True)
lla_dark_fnames, lla_dark_ramps, lla_dark_zodis = \
    lla_dark_fnames[lla_unique], lla_dark_ramps[lla_unique], lla_dark_zodis[lla_unique]

# Make the zodi bins
q = query(setup_superdark.select(DB_foreground.ZODI_12, DB_bcd.CHNLNUM, DB_bcd.RAMPTIME, DB_bcd.MJD_OBS)\
                 .where(((DB_bcd.CHNLNUM==0)|(DB_bcd.CHNLNUM==2))))
zodis, chnlnums, ramptimes, mjds = \
    q['ZODI_12'].to_numpy(), q['CHNLNUM'].to_numpy(), q['RAMPTIME'].to_numpy(), q['MJD_OBS'].to_numpy()

bin_dict = {'SL':{str(r):{} for r in SimlaVar().sl_ramptimes}, 
            'LL':{str(r):{} for r in SimlaVar().ll_ramptimes}, 
            'LLa':{str(r):{} for r in SimlaVar().ll_ramptimes}}

for ramp in SimlaVar().sl_ramptimes:
    
    m = np.where((chnlnums==0)&(np.abs(ramptimes-ramp)<0.01))
    sl_zodis = zodis[m]
    sl_zodi_bin_edges = np.hstack((np.linspace(np.min(sl_zodis), np.max(sl_zodis), 
                                               SimlaVar().n_zodi_bins+1)[:-1], 
                                               [np.inf]))
    sl_fiducial_zodis = np.asarray([np.mean(sl_zodis[np.where((sl_zodis>=sl_zodi_bin_edges[z])&\
                                                           (sl_zodis<sl_zodi_bin_edges[z+1]))]) \
                                 for z in range(0, len(sl_zodi_bin_edges)-1)])

    bin_dict['SL'][str(ramp)]['edges'] = sl_zodi_bin_edges
    bin_dict['SL'][str(ramp)]['fiducials'] = sl_fiducial_zodis

for ramp in SimlaVar().ll_ramptimes:
    
    m = np.where((chnlnums==2)&(np.abs(ramptimes-ramp)<0.01)&(mjds<SimlaVar().ll_gain_change_mjd))
    ll_zodis = zodis[m]
    ll_zodi_bin_edges = np.hstack((np.linspace(np.min(ll_zodis), np.max(ll_zodis), 
                                               SimlaVar().n_zodi_bins+1)[:-1], 
                                               [np.inf]))
    ll_fiducial_zodis = np.asarray([np.mean(ll_zodis[np.where((ll_zodis>=ll_zodi_bin_edges[z])&\
                                                           (ll_zodis<ll_zodi_bin_edges[z+1]))]) \
                                 for z in range(0, len(ll_zodi_bin_edges)-1)])

    bin_dict['LL'][str(ramp)]['edges'] = ll_zodi_bin_edges
    bin_dict['LL'][str(ramp)]['fiducials'] = ll_fiducial_zodis

    # After gain change
    ma = np.where((chnlnums==2)&(np.abs(ramptimes-ramp)<0.01)&(mjds>=SimlaVar().ll_gain_change_mjd))
    lla_zodis = zodis[ma]
    lla_zodi_bin_edges = np.hstack((np.linspace(np.min(lla_zodis), np.max(lla_zodis), 
                                               SimlaVar().n_zodi_bins+1)[:-1], 
                                               [np.inf]))
    lla_fiducial_zodis = np.asarray([np.mean(lla_zodis[np.where((lla_zodis>=lla_zodi_bin_edges[z])&\
                                                           (lla_zodis<lla_zodi_bin_edges[z+1]))]) \
                                 for z in range(0, len(lla_zodi_bin_edges)-1)])

    bin_dict['LLa'][str(ramp)]['edges'] = lla_zodi_bin_edges
    bin_dict['LLa'][str(ramp)]['fiducials'] = lla_fiducial_zodis

pickle.dump(bin_dict, open(simlapath+'storage/superdark_zodi_bin_data.pkl', 'wb'))

# Loop through each ramptime and zodi and make the superdarks
def generate_superdark(masked_file_list):

    '''
    Takes in the list of BCD file names and 
    returns a zodibin-separated list of [superdark, depth_image, superdark_unc]
    where depth_image tells how many BCDs contributed to each pixel.

    '''

    stack = []
    stack_unc = []
    for f in range(len(masked_file_list)):
        
        with fits.open(masked_file_list[f], memmap=False) as hdul:
            imdat = hdul[0].data
            imhead = hdul[0].header
        modname = ['SL', None, 'LL', None][imhead['CHNLNUM']]
        zodiim = np.load(zodi_im_path+str(imhead['AORKEY'])+'_'+modname+'.npy')
        subim = imdat - zodiim

        uncfile = masked_file_list[f].replace('bcd.','func.')
        with fits.open(uncfile, memmap=False) as hdul_unc:
            uncdat = hdul_unc[0].data

        # Do a crude j2-like cut
        n_pass = 0
        for sub in [1, 2]:
            for shardmask in shardmask_lib[imhead['CHNLNUM']][sub-1]:
                if np.nanmedian(subim*shardmask) < med_pixval_cut: n_pass += 1
        if n_pass == [sl_n_shards*2, None, ll_n_shards*2][imhead['CHNLNUM']]:
            stack.append(subim)
            stack_unc.append(uncdat)
            
        if f % 1000 == 0:
            gc.collect()
    stack = np.asarray(stack)
    stack_unc = np.asarray(stack_unc)
    
    if len(stack) > 0:
        trimmed_stack = sigma_clip(stack, maxiters=5, sigma=SimlaVar().sd_trim_sigma, axis=0)
        trimmed_stack = np.where(trimmed_stack=='--', np.nan, trimmed_stack)
        superdark = np.nanmedian(trimmed_stack, axis=0)
        superdark_unc = np.sqrt(np.nansum(stack_unc**2, axis=0)) / np.nansum(stack_unc, axis=0)
        # ^ assuming that the uncertainty of the median is similar to that of the mean
        depth_image = np.nansum(np.where(trimmed_stack==trimmed_stack, 1, 0), axis=0)
        return np.asarray([superdark, depth_image, superdark_unc])
    else: return None

# Create the superdarks for SL and LL
for ramp in tqdm(SimlaVar().sl_ramptimes, desc='generating SL superdarks'):
    zodi_bin_edges = bin_dict['SL'][str(ramp)]['edges']
    fiducial_zodis = bin_dict['SL'][str(ramp)]['fiducials']
    for z in range(0, len(zodi_bin_edges)-1):
        zmin, zmax = zodi_bin_edges[z], zodi_bin_edges[z+1]
        binmask = np.where((np.abs(sl_dark_ramps-ramp)<0.01)&(sl_dark_zodis>=zmin)&(sl_dark_zodis<zmax))
        if len(sl_dark_fnames[binmask]) > 0:
            sd_data = generate_superdark(sl_dark_fnames[binmask].tolist())
            if sd_data is not None:
                np.save(sd_dir+'superdark_SL_ramp-'+str(ramp)+'_fidzodi-'+str(round(fiducial_zodis[z], 1)), sd_data)

for ramp in tqdm(SimlaVar().ll_ramptimes, desc='generating LL superdarks'):
    zodi_bin_edges = bin_dict['LL'][str(ramp)]['edges']
    fiducial_zodis = bin_dict['LL'][str(ramp)]['fiducials']
    for z in range(0, len(zodi_bin_edges)-1):
        zmin, zmax = zodi_bin_edges[z], zodi_bin_edges[z+1]
        binmask = np.where((np.abs(ll_dark_ramps-ramp)<0.01)&(ll_dark_zodis>=zmin)&(ll_dark_zodis<zmax))
        if len(ll_dark_ramps[binmask]) > 0:
            sd_data = generate_superdark(ll_dark_fnames[binmask].tolist())
            if sd_data is not None:
                np.save(sd_dir+'superdark_LL_ramp-'+str(ramp)+'_fidzodi-'+str(round(fiducial_zodis[z], 1)), sd_data)

# LLa refers to LL after the gain change
for ramp in tqdm(SimlaVar().ll_ramptimes, desc='generating LLa superdarks'):
    zodi_bin_edges = bin_dict['LLa'][str(ramp)]['edges']
    fiducial_zodis = bin_dict['LLa'][str(ramp)]['fiducials']
    for z in range(0, len(zodi_bin_edges)-1):
        zmin, zmax = zodi_bin_edges[z], zodi_bin_edges[z+1]
        binmask = np.where((np.abs(lla_dark_ramps-ramp)<0.01)&(lla_dark_zodis>=zmin)&(lla_dark_zodis<zmax))
        if len(lla_dark_fnames[binmask]) > 0:
            sd_data = generate_superdark(lla_dark_fnames[binmask].tolist())
            if sd_data is not None:
                np.save(sd_dir+'superdark_LLa_ramp-'+str(ramp)+'_fidzodi-'+str(round(fiducial_zodis[z], 1)), sd_data)




                