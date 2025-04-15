'''
For each AORKEY, interpolate between the zodi-binned superdarks to create a custom 
superdark that is appropriate for the specific zodi of that AOR. See make_superdarks.py.

The interpolator is in simla_utils.py, so that it can also be called in the validation notebooks.
See load_superdark_sets, interp_superdark in simla_utils.py.

Prerequisite code: bcd_metadata.py, foreground_model.py, make_superdarks.py and prereqs therein.

'''

import os
import glob
import numpy as np
from tqdm import tqdm

from simla_variables import SimlaVar
from simladb import query, setup_superdark, DB_bcd, DB_foreground
from simla_utils import load_superdark_sets, interp_superdark

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

# Get the superdark set
superdark_sets = load_superdark_sets(sd_dir)

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

        ordername = ['SL', 'SH', 'LL', 'LH'][chnl]
        name = str(aor)+'_'+ordername
        np.save(tsd_dir+name, superdark)