from simla.database import query, bcd
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from simla import tools
from scipy import interpolate
import glob
import os
simlapath = os.path.dirname(os.path.realpath(__file__))

# Make BCD-like images for each AORKEY that has the modeled zodiacal light spectrum on them

all_aors = query(bcd.select(bcd.AORKEY))['AORKEY'].to_numpy().tolist()
unique_aors = list(set(all_aors))

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
        
def imprint_spectrum(in_wav, in_spec, header):
    
    import numpy as np
    from astropy.io import fits
    from scipy.interpolate import interp1d

    # Turn an input spectrum into a BCD-like image
    
    chnlnum = header['CHNLNUM']

    if chnlnum < 2:
        conversion = [
        simlapath+'/calib/sl_conversion.fits',
        simlapath+'/calib/sh_conversion.fits'
        ][chnlnum]
        
    elif chnlnum == 2: # handle the gain change in LL
        if header['MJD_OBS'] < 54403.0:
            conversion = simlapath+'/calib/ll_conversion.fits'
        elif header['MJD_OBS'] >= 54403.0:
            conversion = simlapath+'/calib/lla_conversion.fits'
            
    elif chnlnum == 3: # handle the gain change in LH
        if header['MJD_OBS'] < 53653.12:
            conversion = simlapath+'/calib/lh_conversion.fits'
        elif header['MJD_OBS'] >= 53653.12:
            conversion = simlapath+'/calib/lha_conversion.fits'
    conv = fits.getdata(conversion)
    
    contribution_cube = [
        simlapath+'/calib/cc_b0v6.npy',
        simlapath+'/calib/cc_b1v9.npy',
        simlapath+'/calib/cc_b2v10.npy',
        simlapath+'/calib/cc_b3v7.npy',
    ][chnlnum]
    cc = np.load(contribution_cube)
    
    wavsamp_file = [
        '/usr/local/idl/idl_library/cubism/cubism/calib/data/ssc/irs_b0_WAVSAMPv6.tbl',
        '/usr/local/idl/idl_library/cubism/cubism/calib/data/ssc/irs_b1_WAVSAMPv9.tbl',
        '/usr/local/idl/idl_library/cubism/cubism/calib/data/ssc/irs_b2_WAVSAMPv10.tbl',
        '/usr/local/idl/idl_library/cubism/cubism/calib/data/ssc/irs_b3_WAVSAMPv7.tbl',
    ][chnlnum]
    file = open(wavsamp_file, 'r')
    l = file.readlines()
    for line in l:
        if 'int' in line:
            startline = l.index(line)+1
    file.close()
    wavsamp_data = np.genfromtxt(wavsamp_file, skip_header=startline).tolist()
    wavelengths = np.asarray([i[3] for i in wavsamp_data])
    
    # interpolate input spectrum onto wavsamp wavelengths
    s = interp1d(in_wav, in_spec, bounds_error=False, fill_value='extrapolate')
    spectrum = s(wavelengths)
    
    bcd_out = []
    for i in range(len(wavelengths)):
        bcd_out.append(cc[i]*spectrum[i])
    bcd_out = np.nansum(np.asarray(bcd_out), axis=0)
    bcd_out = bcd_out / conv
    
    return bcd_out

zodi_wavs = [np.load('/home/work/simla/zodi_spectra/sl_wavelengths.npy'),
             None,
             np.load('/home/work/simla/zodi_spectra/ll_wavelengths.npy'),
             None]

# Load in the SL CVZ average
sl_cvz_average = np.load(simlapath+'/cvz_campaign_spectra/cvz_average_sl_zodispec.npy')

# Load in campaign data for unrepresented campaigns
campdata = np.load(simlapath+'/IRS_campaigns.npy')
camp_nums = campdata.T[0].astype(float)
camp_names = campdata.T[1]
camp_starts = campdata.T[2].astype(float)
camp_ends = campdata.T[3].astype(float)

if not os.path.exists(simlapath+'/zodi_images/')

for rep in tqdm(bcd_reps):
    
    imhead = fits.getheader(rep)

    zodi_lam = zodi_wavs[imhead['CHNLNUM']]
    modname = ['sl',None,'ll'][imhead['CHNLNUM']]
    zodi_spec = np.load('/home/work/simla/zodi_spectra/'+\
                        str(imhead['AORKEY'])+'_'+modname+'_zodionly.npy')
    
    # Select the appropriate CVZ spectrum which will be subtracted.
    # For SL, this is the CVZ average.
    if imhead['CHNLNUM'] == 0:
        cvz_spec = sl_cvz_average
    # For LL, we need to select the appropriate
    # campaign-dependent CVZ spectrum.
    elif imhead['CHNLNUM'] == 2:
        campaign = imhead['CAMPAIGN']
        cvz_selection = glob.glob(simlapath+'/cvz_campaign_spectra/cvz_for_'+campaign+'.npy')
        if len(cvz_selection) > 0: cvz_spec = np.load(cvz_selection[0])
        else:
            # For some reason, not all campaigns are represented...
            # select the nearest (in time) available spectrum
            t = imhead['MJD_OBS']
            nearest_camp = camp_names[np.where(camp_starts==\
                    camp_starts[np.abs(camp_starts - t).argmin()])][0]
            cvz_spec = np.load(simlapath+'/cvz_campaign_spectra/cvz_for_'+nearest_camp+'.npy')
    zodi_spec = zodi_spec - cvz_spec
    
    zodi_bcd = imprint_spectrum(zodi_lam, zodi_spec, imhead)
    
    name = str(imhead['AORKEY'])+'_'+['SL', 'SH', 'LL', 'LH'][imhead['CHNLNUM']]
    np.save(simlapath+'/zodi_images/'+name, zodi_bcd)