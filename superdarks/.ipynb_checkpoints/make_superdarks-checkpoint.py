import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from tqdm import tqdm
import os
import gc

from simladb import query, setup_superdark, DB_bcd, DB_foreground, DB_judge1
from simla_variables import SimlaVar

# Setup directories
sd_dir = SimlaVar().simlapath+'/superdarks/superdarks/'
if not os.path.exists(sd_dir):
    os.mkdir(sd_dir)

zodi_im_path = SimlaVar().simlapath+'zodi_images/zodi_images/'

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

occurence = np.asarray([np.count_nonzero(dceids==d) for d in tqdm(dceids, desc='counting occurence of dark shards')])

# Fulldark DCEIDs have an occurence = shard_num*2
dark_sl_mask = np.where((chnls==0)&(np.asarray(occurence)==(2*SimlaVar().sl_n_shards)))
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
all_zodi = query(setup_superdark.select(DB_foreground.ZODI_12)\
                 .where(((DB_bcd.CHNLNUM==0)|(DB_bcd.CHNLNUM==2))))['ZODI_12'].to_numpy()
###!!! If you change this, you also need to change it in make_tailored_superdarks.py !!!###
zodi_bin_edges = np.hstack((np.linspace(np.min(all_zodi), np.max(all_zodi), SimlaVar().n_zodi_bins+1)[:-1], 
                           [np.inf]))
# ^use the max val in this sample for spacing, but allow larger values
fiducial_zodis = np.asarray([np.mean(all_zodi[np.where((all_zodi>=zodi_bin_edges[z])&(all_zodi<zodi_bin_edges[z+1]))]) \
                             for z in range(0, len(zodi_bin_edges)-1)])
np.save(SimlaVar().simlapath+'storage/zodi_bin_edges', zodi_bin_edges)
np.save(SimlaVar().simlapath+'storage/fiducial_zodis', fiducial_zodis)

# Loop through each ramptime and zodi and make the superdarks
def generate_superdark(masked_file_list):
    stack = []
    for f in range(len(masked_file_list)):
        with fits.open(masked_file_list[f]) as hdul:
            imdat = hdul[0].data
            imhead = hdul[0].header
        modname = ['SL', None, 'LL', None][imhead['CHNLNUM']]
        zodiim = np.load(zodi_im_path+str(imhead['AORKEY'])+'_'+modname+'.npy')
        subim = imdat - zodiim
        stack.append(subim)
        if f % 1000 == 0:
            gc.collect()
    stack = np.asarray(stack)
    trimmed_stack = sigma_clip(stack, maxiters=5, sigma=SimlaVar().sd_trim_sigma, axis=0)
    trimmed_stack = np.where(trimmed_stack=='--', np.nan, trimmed_stack)
    superdark = np.nanmedian(trimmed_stack, axis=0)
    depth_image = np.nansum(np.where(trimmed_stack==trimmed_stack, 1, 0), axis=0)
    return np.asarray([superdark, depth_image])

for ramp in tqdm(SimlaVar().sl_ramptimes, desc='generating SL superdarks'):
    for z in range(0, len(zodi_bin_edges)-1):
        zmin, zmax = zodi_bin_edges[z], zodi_bin_edges[z+1]
        binmask = np.where((np.abs(sl_dark_ramps-ramp)<0.01)&(sl_dark_zodis>=zmin)&(sl_dark_zodis<zmax))
        if len(sl_dark_fnames[binmask]) > 0:
            sd_data = generate_superdark(sl_dark_fnames[binmask].tolist())
            np.save(sd_dir+'superdark_SL_ramp-'+str(ramp)+'_fidzodi-'+str(round(fiducial_zodis[z], 1)), sd_data)

for ramp in tqdm(SimlaVar().ll_ramptimes, desc='generating LL superdarks'):
    for z in range(0, len(zodi_bin_edges)-1):
        zmin, zmax = zodi_bin_edges[z], zodi_bin_edges[z+1]
        binmask = np.where((np.abs(ll_dark_ramps-ramp)<0.01)&(ll_dark_zodis>=zmin)&(ll_dark_zodis<zmax))
        if len(ll_dark_ramps[binmask]) > 0:
            sd_data = generate_superdark(ll_dark_fnames[binmask].tolist())
            np.save(sd_dir+'superdark_LL_ramp-'+str(ramp)+'_fidzodi-'+str(round(fiducial_zodis[z], 1)), sd_data)

for ramp in tqdm(SimlaVar().ll_ramptimes, desc='generating LLa superdarks'):
    for z in range(0, len(zodi_bin_edges)-1):
        zmin, zmax = zodi_bin_edges[z], zodi_bin_edges[z+1]
        binmask = np.where((np.abs(lla_dark_ramps-ramp)<0.01)&(lla_dark_zodis>=zmin)&(lla_dark_zodis<zmax))
        if len(lla_dark_fnames[binmask]) > 0:
            sd_data = generate_superdark(lla_dark_fnames[binmask].tolist())
            np.save(sd_dir+'superdark_LLa_ramp-'+str(ramp)+'_fidzodi-'+str(round(fiducial_zodis[z], 1)), sd_data)