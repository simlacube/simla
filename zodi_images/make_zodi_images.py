'''
Make IRS BCD-like images that contain the spectrum of pre-computed model zodi
for each AOR. This spectrum is relative to the CVZ spectrum (CVZ-subtracted).
See cvz_campaigns.py

The saved images are in units of e-/s, just like a BCD.

Requires that there is already a directory that contains the zodi spectra for 
each AOR as well as the associated wavelengths.

Make sure that the zodipath is set in simla_variables.py
Prerequisite code: bcd_metadata.py, cvz_campaigns.py.

'''

import numpy as np
from astropy.io import fits
from tqdm import tqdm
from scipy import interpolate
import os

from simla_variables import SimlaVar
from simladb import query, DB_bcd

zodipath = SimlaVar().zodipath
simlapath = SimlaVar().simlapath
storagepath = simlapath+'storage/'

def imprint_spectrum(in_wav, in_spec, conv, cc, wavelengths):

    '''
    Takes an input spectrum and "imprints" it onto a BCD-like image.

    in_wav: array of wavelengths in um
    in_spec: array of zodi fnus in MJy/sr
    conv: the corresponding conversion frame
    cc: the corresponding "contribution cube" that gives the pixel 
        contributions for each WAVSAMP.
    wavelengths: array of the wavelengths associated with the WAVSAMP (um)

    returns the imprinted image in e-/s

    '''
    
    # interpolate input spectrum onto wavsamp wavelengths
    s = interpolate.interp1d(in_wav, in_spec, bounds_error=False, fill_value='extrapolate')
    spectrum = s(wavelengths)

    # Propegate spectrum onto wavsamp and convert
    bcd_out = []
    for i in range(len(wavelengths)):
        bcd_out.append(cc[i]*spectrum[i])
    bcd_out = np.nansum(np.asarray(bcd_out), axis=0)
    bcd_out = bcd_out / conv
    
    return bcd_out

# Load in all necessary calibration files
sl_conv, ll_conv, lla_conv = \
    fits.getdata(simlapath+'calib/conversion_frames/sl_conversion.fits'), \
    fits.getdata(simlapath+'calib/conversion_frames/ll_conversion.fits'), \
    fits.getdata(simlapath+'calib/conversion_frames/lla_conversion.fits')

sl_cc, ll_cc = \
    np.load(simlapath+'calib/contribution_cubes/cc_b0v6.npy'), \
    np.load(simlapath+'calib/contribution_cubes/cc_b2v10.npy'), \

# This blocl interprets the WAVSAMP files and gets the wavelengths
def prep_wavsamp(f):
    wdat = open(f, 'r')
    l = wdat.readlines()
    for line in l:
        if 'int' in line:
            startline = l.index(line)+1
    wdat.close()
    wavsamp_data = np.genfromtxt(f, skip_header=startline).tolist()
    wavelengths = np.asarray([i[3] for i in wavsamp_data])
    return wavelengths
sl_lam, ll_lam = \
    prep_wavsamp(simlapath+'calib/wavsamp/irs_b0_WAVSAMPv6.tbl'), \
    prep_wavsamp(simlapath+'calib/wavsamp/irs_b2_WAVSAMPv10.tbl')

zodi_wavs = [np.load(zodipath+'sl_wavelengths.npy'),
             None,
             np.load(zodipath+'ll_wavelengths.npy'),
             None]

# Query database for necessary BCD information
q = query(DB_bcd.select(DB_bcd.AORKEY, DB_bcd.CHNLNUM, DB_bcd.MJD_OBS, DB_bcd.CAMPAIGN, DB_bcd.RAMPTIME)\
         .where(((DB_bcd.CHNLNUM==0)|(DB_bcd.CHNLNUM==2))))
aorkeys, chnlnums, mjds, campaigns, ramptimes = \
    q['AORKEY'].to_numpy(), q['CHNLNUM'].to_numpy(), \
    q['MJD_OBS'].to_numpy(), q['CAMPAIGN'].to_numpy(), q['RAMPTIME'].to_numpy()

# Set up directory structure
if not os.path.exists(simlapath+'zodi_images/zodi_images/'):
    os.mkdir(simlapath+'zodi_images/zodi_images/')

for aor in tqdm(np.unique(aorkeys)):

    existing_channels = np.unique(chnlnums[np.where(aorkeys==aor)])

    for chnl in existing_channels:

        index = np.where((aorkeys==aor)&(chnlnums==chnl))[0][0]
        ramp, camp, mjd = ramptimes[index], campaigns[index], mjds[index]
            
        zodi_lam = zodi_wavs[chnl]
        modname = ['sl',None,'ll'][chnl]
        zodi_spec = np.load(zodipath+str(aor)+'_'+modname+'_zodionly.npy')
        
        # Select the appropriate CVZ spectrum which will be subtracted.
        cvz_spec = np.load(storagepath+'cvz_campaign_spectra/'+'CVZ_'+modname+'_'+str(ramp)+'_'+camp+'.npy')
        
        zodi_spec = zodi_spec - cvz_spec

        if chnl == 0:
            conv, cc, wavelengths = sl_conv, sl_cc, sl_lam
        elif chnl == 2:
            cc, wavelengths = ll_cc, ll_lam
            if mjd < 54403.0: conv = ll_conv 
            else: conv = lla_conv # gain change in LL
                
        zodi_bcd = imprint_spectrum(zodi_lam, zodi_spec, conv, cc, wavelengths)
        
        name = str(aor)+'_'+['SL', 'SH', 'LL', 'LH'][chnl]
        np.save(simlapath+'zodi_images/zodi_images/'+name, zodi_bcd)