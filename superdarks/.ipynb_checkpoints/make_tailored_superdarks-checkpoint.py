import os
import glob
import numpy as np
from tqdm import tqdm

from simla_variables import SimlaVar
from simladb import query, setup_superdark, DB_bcd, DB_foreground

# Setup directories
sd_dir = SimlaVar().simlapath+'superdarks/'
tsd_dir = sd_dir+'tailored_superdarks/'
if not os.path.exists(tsd_dir):
    os.mkdir(tsd_dir)

# Query database for necessary BCD information
q = query(setup_superdark.select(DB_bcd.AORKEY, DB_bcd.CHNLNUM, DB_bcd.MJD_OBS, DB_bcd.RAMPTIME, DB_foreground.ZODI_12)\
         .where(((DB_bcd.CHNLNUM==0)|(DB_bcd.CHNLNUM==2))))
aorkeys, chnlnums, mjds, ramptimes, zodis = \
    q['AORKEY'].to_numpy(), q['CHNLNUM'].to_numpy(), \
    q['MJD_OBS'].to_numpy(), q['RAMPTIME'].to_numpy(), q['ZODI_12'].to_numpy()

# Load in fiducial superdark data
# If a zodi bin is missing, we just interpolate over it
superdark_sets = {}
for ramp in SimlaVar().sl_ramptimes:
    ramp_superdarks = sorted(glob.glob(sd_dir+'superdarks/*SL*'+str(ramp)+'*'))
    fiducial_zodis = np.asarray(sorted([float(i.split('fidzodi')[-1].split('_')[0].split('.npy')[0]) for i in ramp_superdarks]))
    superdark_sets['SL_'+str(ramp)] = {'set': np.asarray([np.load(i)[0] for i in ramp_superdarks]),
                                      'fiducial_zodis': fiducial_zodis}
    
for ramp in SimlaVar().ll_ramptimes:
    ramp_superdarks = sorted(glob.glob(sd_dir+'superdarks/*LL_*'+str(ramp)+'*'))
    fiducial_zodis = np.asarray(sorted([float(i.split('fidzodi')[-1].split('_')[0].split('.npy')[0]) for i in ramp_superdarks]))
    superdark_sets['LL_'+str(ramp)] = {'set': np.asarray([np.load(i)[0] for i in ramp_superdarks]),
                                      'fiducial_zodis': fiducial_zodis}

for ramp in SimlaVar().ll_ramptimes:
    ramp_superdarks = sorted(glob.glob(sd_dir+'superdarks/*LLa*'+str(ramp)+'*'))
    fiducial_zodis = np.asarray(sorted([float(i.split('fidzodi')[-1].split('_')[0].split('.npy')[0]) for i in ramp_superdarks]))
    superdark_sets['LLa_'+str(ramp)] = {'set': np.asarray([np.load(i)[0] for i in ramp_superdarks]),
                                       'fiducial_zodis': fiducial_zodis}

# Interpolator function for tailored superdarks
def interp_superdark(z, superdark_set):
    fiducial_zodis = superdark_set['fiducial_zodis']
    superdark_set = superdark_set['set']
    if z < np.min(fiducial_zodis): z = np.min(fiducial_zodis)
    if z > np.max(fiducial_zodis): z = np.max(fiducial_zodis)
    if len(fiducial_zodis[z==fiducial_zodis]) > 0:
        out_image = superdark_set[z==fiducial_zodis][0]
    else:
        z_low = np.max(fiducial_zodis[fiducial_zodis<z])
        z_high = np.min(fiducial_zodis[fiducial_zodis>z])
        z_diff = z_high - z_low
        low_weight = (z_high - z)/z_diff
        high_weight = (z - z_low)/z_diff
        low_im = superdark_set[fiducial_zodis==z_low][0]
        high_im = superdark_set[fiducial_zodis==z_high][0]
        out_image = np.average((low_im, high_im), axis=0, weights=(low_weight, high_weight))
    return out_image

# Make the tailored superdarks
for aor in tqdm(np.unique(aorkeys), desc='making tailored superdarks for each AOR'):

    existing_channels = np.unique(chnlnums[np.where(aorkeys==aor)])

    for chnl in existing_channels:

        index = np.where((aorkeys==aor)&(chnlnums==chnl))[0][0]
        ramp, mjd, zodi = ramptimes[index], mjds[index], zodis[index]
    
        ordername = ['SL', 'SH', 'LL', 'LH'][chnl]
        if chnl == 2 and mjd >= SimlaVar().ll_gain_change_mjd:
            ordername = 'LLa'
        else: ordername = ordername
    
        superdark_set = superdark_sets[ordername+'_'+str(ramp)]
        superdark = interp_superdark(zodi, superdark_set)
        
        name = str(aor)+'_'+ordername
        np.save(tsd_dir+name, superdark)