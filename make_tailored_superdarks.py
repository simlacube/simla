from simla.database import query, bcd, zodi
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from simla import tools
from scipy import interpolate
import glob
import os
simlapath = os.path.dirname(os.path.realpath(__file__))

all_aors = query(bcd.select(bcd.AORKEY))['AORKEY'].to_numpy().tolist()
unique_aors = list(set(all_aors))

ll_gain_change_mjd = 54403.0

bcd_reps = []
for aor in tqdm(unique_aors):
    
    sl_bcds = query(bcd.select(bcd.FILE_NAME).where(bcd.AORKEY==aor).where(bcd.CHNLNUM==0))['FILE_NAME'].to_numpy().tolist()
    if len(sl_bcds) > 0:
        sl_bcdfile = sl_bcds[0]
        bcd_reps.append(sl_bcdfile)
        
    ll_bcds = query(bcd.select(bcd.FILE_NAME).where(bcd.AORKEY==aor).where(bcd.CHNLNUM==2))['FILE_NAME'].to_numpy().tolist()
    if len(ll_bcds) > 0:    
        ll_bcdfile = ll_bcds[0]
        bcd_reps.append(ll_bcdfile)
    
sl_ramptimes = [6.29, 14.68, 241.83, 60.95]
ll_ramptimes = [6.29, 14.68, 31.46, 121.90]
superdark_sets = {}
for i in sl_ramptimes:
    
    superdark_set_files = sorted(glob.glob(simlapath+'/superdarks/SL*'+str(i)+'*'))
    superdark_set = np.asarray([np.load(i) for i in superdark_set_files])
    fiducial_zodis = np.asarray([float(i.split('superdark_')[1].split('-')[0])+5 for i in superdark_set_files])
    superdark_sets['SL_'+str(i)] = {'set': superdark_set, 'fiducial_zodis': fiducial_zodis}
    
for i in ll_ramptimes:
    
    superdark_set_files = sorted(glob.glob(simlapath+'/superdarks/LL*'+str(i)+'*'))
    superdark_set = np.asarray([np.load(i) for i in superdark_set_files])
    fiducial_zodis = np.asarray([float(i.split('superdark_')[1].split('-')[0])+5 for i in superdark_set_files])
    superdark_sets['LL_'+str(i)] = {'set': superdark_set, 'fiducial_zodis': fiducial_zodis}
    
    superdark_set_files = sorted(glob.glob(simlapath+'/superdarks/LLa*'+str(i)+'*'))
    superdark_set = np.asarray([np.load(i) for i in superdark_set_files])
    fiducial_zodis = np.asarray([float(i.split('superdark_')[1].split('-')[0])+5 for i in superdark_set_files])
    superdark_sets['LLa_'+str(i)] = {'set': superdark_set, 'fiducial_zodis': fiducial_zodis}

# function for interpolating between superdarks
def interp_superdark(z, superdark_set, fiducial_zodis):
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
    # out_image = ndimage.median_filter(out_image, size=3)
    # out_image = np.where((out_image>-15)&(out_image<15), out_image, 0)
    return out_image

if not os.path.exists(simlapath+'/')

for rep in tqdm(bcd_reps):
    
    imhead = fits.getheader(rep)
    aorkey = imhead['AORKEY']
    
    ordername = ['SL', 'SH', 'LL', 'LH'][imhead['CHNLNUM']]
    if imhead['CHNLNUM'] == 2 and imhead['MJD_OBS'] >= ll_gain_change_mjd:
        sd_ordername = 'LLa'
    else: sd_ordername = ordername
    
    superdark_set = superdark_sets[sd_ordername+'_'+str(imhead['RAMPTIME'])]['set']
    fiducial_zodis = superdark_sets[sd_ordername+'_'+str(imhead['RAMPTIME'])]['fiducial_zodis']
    
    zodi_value = query(zodi.select(zodi.LL1_C).where(zodi.AORKEY==str(aorkey)))['LL1_C'][0]
    superdark = interp_superdark(zodi_value, superdark_set, fiducial_zodis)
    
    name = str(aorkey)+'_'+ordername
    np.save(simlapath+'/tailored_superdarks/'+name, superdark)