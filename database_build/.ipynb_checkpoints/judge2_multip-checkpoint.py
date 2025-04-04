import numpy as np
from astropy.io import fits
from tqdm import tqdm
from mysql.connector import connect
from multiprocessing import Pool
import os

from simla_variables import SimlaVar
from simladb import query, DB_bcd
from simla_utils import bcd_spectrum

### MULTIPROCESSING VERSION ###

def get_specdata_and_enter(i):

    dceid, aor, fname, chnl, mjd = \
        dceids[i], aors[i], fnames[i], chnls[i], mjds[i]

    with fits.open(fname) as hdul:
        header = hdul[0].header
        image_data = hdul[0].data

    # Mask known bad pixels
    image_data = np.where((fits.getdata((fname.replace('bcd.','bmask.')))\
                                     &28928 !=0).astype('int'), np.nan, image_data)

    # Load in the zodi image and the superdark
    ordername = ['SL', 'SH', 'LL', 'LH'][chnl]
    zodiim = np.load(zodi_im_path+str(aor)+'_'+ordername+'.npy')
    if chnl == 2 and mjd >= ll_gain_change_mjd:
        ordername = 'LLa'
    else: ordername = ordername
    superdark = np.load(tsd_dir+str(aor)+'_'+ordername+'.npy')

    # Get the BCD in zodi and superdark - free space
    subim = (image_data - zodiim) - superdark

    # Extract data for all shards
    shardnum = [sl_shardnum, None, ll_shardnum][chnl]
    for n in range(shardnum):

        l1, f1 = extractor.subslit_bcd_spectrum(subim, header, 1, n)
        l2, f2 = extractor.subslit_bcd_spectrum(subim, header, 2, n)

        f1_median = np.nanmedian(f1)
        f2_median = np.nanmedian(f2)

        f1_mean = np.nanmean(f1)
        f2_mean = np.nanmean(f2)

        f1_stdev = np.nanstd(f1)
        f2_stdev = np.nanstd(f2)

        bl1, bf1 = bin_spectrum(l1, f1, nbins)
        bl2, bf2 = bin_spectrum(l2, f2, nbins)

        # Upload to the database
        ADD1 = f"""
            INSERT INTO judge2 (DCEID, CHNLNUM, SUBORDER, SHARD,
            SPEC0, SPEC1, SPEC2, SPEC3, SPEC4, SPEC5, SPEC6, SPEC7, SPEC8, SPEC9,
            F_MEDIAN, F_MEAN, F_STDEV) 
            VALUES ({dceid}, {chnl}, {1}, {n}, 
            {bf1[0]}, {bf1[1]}, {bf1[2]}, {bf1[3]}, {bf1[4]}, {bf1[5]}, {bf1[6]}, {bf1[7]}, {bf1[8]}, {bf1[9]},
            {f1_median}, {f1_mean}, {f1_stdev});
            """
        cursor.execute(ADD1)
        connection.commit()

        ADD2 = f"""
            INSERT INTO judge2 (DCEID, CHNLNUM, SUBORDER, SHARD,
            SPEC0, SPEC1, SPEC2, SPEC3, SPEC4, SPEC5, SPEC6, SPEC7, SPEC8, SPEC9,
            F_MEDIAN, F_MEAN, F_STDEV) 
            VALUES ({dceid}, {chnl}, {2}, {n}, 
            {bf2[0]}, {bf2[1]}, {bf2[2]}, {bf2[3]}, {bf2[4]}, {bf2[5]}, {bf2[6]}, {bf2[7]}, {bf2[8]}, {bf2[9]},
            {f2_median}, {f2_mean}, {f2_stdev});
            """
        cursor.execute(ADD2)
        connection.commit()

    if chnl == 0:
        bl_sl1, bl_sl2 = bl1, bl2
    elif chnl == 2:
        bl_ll1, bl_ll2 = bl1, bl2
    
# Bin the extracted spectra by wavelength
def bin_spectrum(lam, spec, nbins):
    bins = np.histogram(lam, nbins)[-1]
    digits = np.digitize(lam, bins)-1
    digits = np.where(digits==nbins, nbins-1, digits)
    binned_wavelengths = [[] for i in range(nbins)]
    binned_spectrum = [[] for i in range(nbins)]
    for i in range(len(digits)):
        binned_wavelengths[digits[i]].append(lam[i])
        binned_spectrum[digits[i]].append(spec[i])
    bin_lam = np.asarray([np.median(i) for i in binned_wavelengths])
    bin_spec= np.asarray([np.median(i) for i in binned_spectrum])
    return bin_lam, bin_spec
### Not in simla_variables because the DB table would need to be changed as well
nbins = 10
###

ll_gain_change_mjd = SimlaVar().ll_gain_change_mjd
sl_shardnum, ll_shardnum = SimlaVar().sl_n_shards, SimlaVar().ll_n_shards

irspath = SimlaVar().irspath
tsd_dir = SimlaVar().simlapath+'superdarks/tailored_superdarks/'
zodi_im_path = SimlaVar().simlapath+'zodi_images/zodi_images/'

connection = connect(host="localhost",user="root",database="SIMLA")
cursor = connection.cursor()

# Get all SL and LL BCDs
q = query(DB_bcd.select(DB_bcd.DCEID, DB_bcd.AORKEY, DB_bcd.FILE_NAME, DB_bcd.CHNLNUM, DB_bcd.MJD_OBS) \
             .where((DB_bcd.CHNLNUM==0)|(DB_bcd.CHNLNUM==2)))
dceids, aors, fnames, chnls, mjds = \
    q['DCEID'].to_numpy(), q['AORKEY'].to_numpy(), irspath+q['FILE_NAME'].to_numpy(), \
    q['CHNLNUM'].to_numpy(), q['MJD_OBS'].to_numpy()

# Initialize the spectrum extractor
extractor = bcd_spectrum()

# Save bin wavelengths for later
bl_sl1, bl_sl2, bl_ll1, bl_ll2 = \
    None, None, None, None

# Loop through BCDs, subtract the zodi and superdark, and upload to database
i_s = range(len(dceids))
if __name__ == '__main__':
    with Pool(processes=8) as pool: 
        for _ in tqdm(pool.imap_unordered(get_specdata_and_enter, i_s), \
                           total=len(i_s)):
            pass

storagepath = SimlaVar().simlapath+'storage/'
np.save(storagepath+'sl1_binlam', bl_sl1)
np.save(storagepath+'sl2_binlam', bl_sl2)
np.save(storagepath+'ll1_binlam', bl_ll1)
np.save(storagepath+'ll2_binlam', bl_ll2)












    