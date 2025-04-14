'''
Make model zodi spectra for the CVZ averaged together in the same way
as the IRS calibration darks were averaged. The result is a zodi spectrum
for each campaign+RAMPTIME combo for accurate zodi subtraction.

We can't know exactly how these were averaged, so we get scale factors empirically.

Make sure the irspath is set in simla_variables.
Prerequisite code: bcd_metadata.py

'''

import numpy as np
from tqdm import tqdm
import os
from scipy import interpolate
from astropy.io import fits
import glob

from simla_variables import SimlaVar
from simladb import query, DB_bcd
from simla_utils import bcd_spectrum

zodipath = SimlaVar().zodipath
simlapath = SimlaVar().simlapath
irspath = SimlaVar().irspath
storagepath = simlapath+'storage/'

# Query for info on CVZ observations and organize 
search = query(DB_bcd.select(DB_bcd.FILE_NAME, DB_bcd.AORKEY, DB_bcd.CAMPAIGN, DB_bcd.CHNLNUM, DB_bcd.RAMPTIME)\
               .where(((DB_bcd.CHNLNUM==0)|(DB_bcd.CHNLNUM==2)) & (DB_bcd.OBJECT=='NCVZ-dark-spot')))
fnames = search['FILE_NAME'].to_numpy()
aorkeys = search['AORKEY'].to_numpy()
campnames = search['CAMPAIGN'].to_numpy()
chnlnums = search['CHNLNUM'].to_numpy()
ramptimes = search['RAMPTIME'].to_numpy()

fnames = np.asarray([irspath+i for i in fnames])

if not os.path.exists(storagepath+'/cvz_campaign_spectra/'):
    os.mkdir(storagepath+'/cvz_campaign_spectra/')

# Load in model wavelengths for both orders and specify
# a mask for normalization
sl_norm_lam = 10.95 # Order center of SL1
ll_norm_lam = 28.75 # Order center of LL1
sl_zodi_lam = np.load(zodipath+'sl_wavelengths.npy')
ll_zodi_lam = np.load(zodipath+'ll_wavelengths.npy')
sl_norm_mask = np.where((sl_zodi_lam>=sl_norm_lam-0.5)&(sl_zodi_lam<=sl_norm_lam+0.5))
ll_norm_mask = np.where((ll_zodi_lam>=ll_norm_lam-0.5)&(ll_zodi_lam<=ll_norm_lam+0.5))

# For both SL and LL, make a normalized CVZ-zodi model
sl_cvz_zodis = np.asarray([np.load(zodipath+str(aor)+'_sl_zodionly.npy') \
                    for aor in aorkeys[np.where(chnlnums==0)]])
sl_cvz_zodi_avspec = np.nanmedian(sl_cvz_zodis, axis=0)
norm_sl_avspec = sl_cvz_zodi_avspec / np.nanmedian(sl_cvz_zodi_avspec[sl_norm_mask])
np.save(storagepath+'/cvz_campaign_spectra/sl_zodi_normspec', norm_sl_avspec)

ll_cvz_zodis = np.asarray([np.load(zodipath+str(aor)+'_ll_zodionly.npy') \
                    for aor in aorkeys[np.where(chnlnums==2)]])
ll_cvz_zodi_avspec = np.nanmedian(ll_cvz_zodis, axis=0)
norm_ll_avspec = ll_cvz_zodi_avspec / np.nanmedian(ll_cvz_zodi_avspec[ll_norm_mask])
np.save(storagepath+'/cvz_campaign_spectra/ll_zodi_normspec', norm_ll_avspec)

# Make dictionaries for the scaler data to be stored in
unique_campnames = np.unique(campnames)
unique_sl_ramptimes = np.unique(ramptimes[np.where(chnlnums==0)])
unique_ll_ramptimes = np.unique(ramptimes[np.where(chnlnums==2)])
sl_scalers = {camp: {ramp: [] for ramp in unique_sl_ramptimes} \
                  for camp in unique_campnames}
ll_scalers = {camp: {ramp: [] for ramp in unique_ll_ramptimes} \
                  for camp in unique_campnames}

# Set up the mask to select the fluxes for scaling using "sample" bcds
extractor = bcd_spectrum()

sl_sample = fnames[np.where(chnlnums==0)][0]
sl_obslams = extractor.fullslit_bcd_spectrum(fits.getdata(sl_sample), fits.getheader(sl_sample), 1)[0]
sl_scale_mask = np.where((sl_obslams>=sl_norm_lam-0.5)&(sl_obslams<=sl_norm_lam+0.5))

ll_sample = fnames[np.where(chnlnums==2)][0]
ll_obslams = extractor.fullslit_bcd_spectrum(fits.getdata(ll_sample), fits.getheader(ll_sample), 1)[0]
ll_scale_mask = np.where((ll_obslams>=ll_norm_lam-0.5)&(ll_obslams<=ll_norm_lam+0.5))

## For SL, all reference darks come from the same set (but separated by RAMPTIME).
# That's what they say, but I am finding that if you have a single dark per RAMPTIME,
# There begins to be a disagreement later in the mission, so I just make them
# campaign-dependent anyway.
sl_fname_mask = np.where(chnlnums==0)
for fname in tqdm(fnames[sl_fname_mask]):

    imdat = fits.getdata(fname)
    imhead = fits.getheader(fname)

    zodi_lam = sl_zodi_lam
    scale_mask = sl_scale_mask

    # Extract the BCD spectrum
    l, f = extractor.fullslit_bcd_spectrum(imdat, imhead, 1)

    # Load in the model spectrum
    zodi_spec = np.load(zodipath+str(imhead['AORKEY'])+'_sl_zodionly.npy')
    fz = interpolate.interp1d(zodi_lam, zodi_spec, bounds_error=False, fill_value=np.nan)(l)

    # Find the difference (scale factor)
    scale = np.nanmedian(fz[scale_mask]) - np.nanmedian(f[scale_mask])

    sl_scalers[imhead['CAMPAIGN']][imhead['RAMPTIME']].append(scale)

# For each campaign, order, and RAMPTIME, median combine these scalers
# and apply to the normalized spectra, and save.
for campname in unique_campnames:

    for ramp in sl_scalers[campname].keys():
        if len(sl_scalers[campname][ramp]) > 0:
            av_scale = np.nanmedian(sl_scalers[campname][ramp])
            scaled_spec = av_scale*norm_sl_avspec
            np.save(storagepath+'/cvz_campaign_spectra/'+'CVZ_sl_'+str(ramp)+'_'+campname, scaled_spec)

## For LL, darks are campaign-dependent
# For each campaign, and each RAMPTIME, loop through all corresponding CVZ BCDs and find the 
# average scale factor for the normalized spectrum. This scale factor is 
# based on the difference between the original zodi model for a CVZ BCD and 
# what is observed in the data.
ll_fname_mask = np.where(chnlnums==2)
for fname in tqdm(fnames[ll_fname_mask]):

    imdat = fits.getdata(fname)
    imhead = fits.getheader(fname)

    zodi_lam = ll_zodi_lam
    scale_mask = ll_scale_mask

    # Extract the BCD spectrum
    l, f = extractor.fullslit_bcd_spectrum(imdat, imhead, 1)

    # Load in the model spectrum
    zodi_spec = np.load(zodipath+str(imhead['AORKEY'])+'_ll_zodionly.npy')
    fz = interpolate.interp1d(zodi_lam, zodi_spec, bounds_error=False, fill_value=np.nan)(l)

    # Find the difference (scale factor)
    scale = np.nanmedian(fz[scale_mask]) - np.nanmedian(f[scale_mask])

    ll_scalers[imhead['CAMPAIGN']][imhead['RAMPTIME']].append(scale)

# For each campaign, order, and RAMPTIME, median combine these scalers
# and apply to the normalized spectra, and save.
for campname in unique_campnames:

    for ramp in ll_scalers[campname].keys():
        if len(ll_scalers[campname][ramp]) > 0:
            av_scale = np.nanmedian(ll_scalers[campname][ramp])
            scaled_spec = av_scale*norm_ll_avspec
            np.save(storagepath+'/cvz_campaign_spectra/'+'CVZ_ll_'+str(ramp)+'_'+campname, scaled_spec)

# There are combinations of campaigns and ramptimes that exist in the non-CVZ data
# but are not represented by CVZ data. For these, average the computed spectra of the 
# two existing CVZ spectra surrounding it

# Load in campaign data for unrepresented campaigns
all_campaign_data = query(DB_bcd.select(DB_bcd.CAMPAIGN, DB_bcd.MJD_OBS).where((DB_bcd.CHNLNUM==0)|(DB_bcd.CHNLNUM==2)))
all_campaigns, all_mjds = all_campaign_data['CAMPAIGN'].to_numpy(), all_campaign_data['MJD_OBS'].to_numpy()
camp_names = np.unique(all_campaigns)
camp_midtimes = np.asarray([np.mean(all_mjds[np.where(all_campaigns==camp)]) for camp in camp_names])

# SL
for ramp in unique_sl_ramptimes:
    
    existing_campnames = np.asarray([i.split('_')[-1][:-4] \
                for i in glob.glob(storagepath+'/cvz_campaign_spectra/'+'CVZ_sl_'+str(ramp)+'_*.npy')])
    
    existing_midtimes = []
    for campname in existing_campnames:
        existing_midtimes.append(camp_midtimes[np.where(camp_names==campname)][0])
    existing_midtimes = np.asarray(existing_midtimes)
    
    for campname in camp_names:
        if campname not in existing_campnames:
            
            this_camptime = camp_midtimes[np.where(camp_names==campname)][0]
            
            camp_before = existing_campnames[np.where(existing_midtimes==np.max(existing_midtimes\
                                                [np.where(existing_midtimes<this_camptime)]))][0]
            camp_after = existing_campnames[np.where(existing_midtimes==np.min(existing_midtimes\
                                                [np.where(existing_midtimes>this_camptime)]))][0]
            
            cvz_before = np.load(storagepath+'/cvz_campaign_spectra/'+'CVZ_sl_'+str(ramp)+'_'+camp_before+'.npy')
            cvz_after = np.load(storagepath+'/cvz_campaign_spectra/'+'CVZ_sl_'+str(ramp)+'_'+camp_after+'.npy')
            new_cvz = np.mean([cvz_before, cvz_after], axis=0)
            np.save(storagepath+'/cvz_campaign_spectra/'+'CVZ_sl_'+str(ramp)+'_'+campname, new_cvz)

# LL
for ramp in unique_ll_ramptimes:
    
    existing_campnames = np.asarray([i.split('_')[-1][:-4] \
                for i in glob.glob(storagepath+'/cvz_campaign_spectra/'+'CVZ_ll_'+str(ramp)+'_*.npy')])
    
    existing_midtimes = []
    for campname in existing_campnames:
        existing_midtimes.append(camp_midtimes[np.where(camp_names==campname)][0])
    existing_midtimes = np.asarray(existing_midtimes)
    
    for campname in camp_names:
        if campname not in existing_campnames:
            
            this_camptime = camp_midtimes[np.where(camp_names==campname)][0]
            
            camp_before = existing_campnames[np.where(existing_midtimes==np.max(existing_midtimes\
                                                [np.where(existing_midtimes<this_camptime)]))][0]
            camp_after = existing_campnames[np.where(existing_midtimes==np.min(existing_midtimes\
                                                [np.where(existing_midtimes>this_camptime)]))][0]
            
            cvz_before = np.load(storagepath+'/cvz_campaign_spectra/'+'CVZ_ll_'+str(ramp)+'_'+camp_before+'.npy')
            cvz_after = np.load(storagepath+'/cvz_campaign_spectra/'+'CVZ_ll_'+str(ramp)+'_'+camp_after+'.npy')
            new_cvz = np.mean([cvz_before, cvz_after], axis=0)
            np.save(storagepath+'/cvz_campaign_spectra/'+'CVZ_ll_'+str(ramp)+'_'+campname, new_cvz)


    

