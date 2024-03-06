from simla.database import query, bcd, qAll, shardphot, galcoords, zodi, qShard
import numpy as np
import scipy
from astropy.io import fits
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from simla import tools
from scipy import interpolate
import os

# Make the superdarks binned by zodi and ramptime that the tailored superdarks
# will be interpolated from

# On-sky cuts to exclude very bright areas for Judge 1
orion_Lmax, orion_Lmin, orion_Bmax, orion_Bmin = 215, 203, -14, -24
cygloop_Lmax, cygloop_Lmin, cygloop_Bmax, cygloop_Bmin = 354.0, 324.5, 31.0, -9.2
lat_cut = 1.5

# Cuts for J1 (MJy/sr)
# J1_low = -0.05
# J1_high = 0.05
J1_low = -0.5
J1_high = 0.5

ll_gain_change_mjd = 54403.0

def get_dark_bcds(time_low, time_high, chnlnum, zodi_cut_low, zodi_cut_high, ramp):
    # return dark BCDs within the desired zodi range

    # Start by selecting all BCDs within the desired zodi range
    selected_bcds = query(qAll.select(bcd.FILE_NAME, bcd.DCEID)\
                     .where(bcd.CHNLNUM==chnlnum)\
                     .where(((galcoords.GALACTIC_B > lat_cut)|
                        (galcoords.GALACTIC_B < -lat_cut))&
                        ((galcoords.GALACTIC_L < orion_Lmin)|
                        (galcoords.GALACTIC_L > orion_Lmax)|
                        (galcoords.GALACTIC_B < orion_Bmin)|
                        (galcoords.GALACTIC_B > orion_Bmax))&
                        ((galcoords.GALACTIC_L < cygloop_Lmin)|
                        (galcoords.GALACTIC_L > cygloop_Lmax)|
                        (galcoords.GALACTIC_B < cygloop_Bmin)|
                        (galcoords.GALACTIC_B > cygloop_Bmax)))
                     .where(bcd.OBJTYPE!='TargetMovingSingle')\
                     .where(bcd.OBJECT!='NCVZ-dark-spot')\
                     .where(zodi.LL1_C>=zodi_cut_low).where(zodi.LL1_C<=zodi_cut_high)\
                     .where((bcd.RAMPTIME>ramp-0.01)&(bcd.RAMPTIME<ramp+0.01))\
                     .where((bcd.MJD_OBS>=time_low)&(bcd.MJD_OBS<time_high)).distinct())

    fnames, dceids = selected_bcds['FILE_NAME'].to_numpy(), \
                     selected_bcds['DCEID'].to_numpy()

    # Uncomment these if you want to use only 1% of selected darks (for testing reasons)
    # ########################
    # fnames = fnames[0::100]
    # dceids = dceids[0::100]
    # ########################

    # Now we have to loop through each of these to find the BCDs on which all 12
    # shards qualify as darks. 
    # The obvious inefficiency of this is owing to an unfortunate database issue.
    dark_selection = []
    for i in tqdm(range(len(dceids))):
        darks = query(qShard.select(shardphot.SUBORDER, shardphot.SUBSLIT)\
             .where(shardphot.CHNLNUM==chnlnum)\
             .where(shardphot.DCEID==dceids[i])\
             .where(shardphot.BACKSUB_PHOT!=0.0)
             .where((shardphot.BACKSUB_PHOT<=J1_high)&(shardphot.BACKSUB_PHOT>=J1_low)))

        suborders = darks['SUBORDER'].to_numpy()
        subslits = darks['SUBSLIT'].to_numpy()

        # See whether all 12 shards are dark
        o1 = subslits[np.where(suborders==1)]
        o2 = subslits[np.where(suborders==2)]
        if len(o1)==6 and len(o2)==6:
            dark_selection.append(fnames[i])
    
    return dark_selection

def superdark(darks):
    
    def iterative_trim(stack, iters=1, stdev=3):
        for i in range(iters):
            std_dev_im = np.nanstd(stack, axis=0)
            mean_im = np.nanmean(stack, axis=0)
            stack = np.where( (stack>mean_im+(stdev*std_dev_im)) | \
                             (stack<mean_im-(stdev*std_dev_im)), np.nan, stack)
        return stack

    subtracted_images = []
    for dark in tqdm(darks):
        
        with fits.open(dark) as hdul:
            irs_imdat = hdul[0].data
            irs_imhead = hdul[0].header
        hdul.close()
        
        modname = ['SL', None, 'LL', None][irs_imhead['CHNLNUM']]
        aorkey = irs_imhead['AORKEY']
        
        zodi_bcd = np.load('./zodi_images/'+\
                          str(aorkey)+'_'+modname+'.npy')
        
        sub_im = irs_imdat - zodi_bcd
        
        subtracted_images.append(sub_im)
        
    subtracted_images = np.asarray(subtracted_images)
    trimmed_stack = iterative_trim(subtracted_images, iters=5, stdev=10)
    stacked_image = np.nanmedian(trimmed_stack, axis=0)

    return stacked_image

zodi_cuts = np.hstack((np.arange(0, 60, 10),np.asarray([60])))
sl_ramptimes = [6.29, 14.68, 241.83, 60.95]
ll_ramptimes = [6.29, 14.68, 31.46, 121.90]

if not os.path.exists('./superdarks/'):
    os.mkdir('./superdarks/')

for ramp in sl_ramptimes:
    for i in range(len(zodi_cuts)-1):

        darks = get_dark_bcds(0, 55000, 0, zodi_cuts[i], zodi_cuts[i+1], ramp)
        if len(darks) > 0:
            darkstack = superdark(darks)
            np.save('./superdarks/SL_superdark_'+\
                    str(zodi_cuts[i])+'-'+str(zodi_cuts[i+1])+'_ramp='+str(ramp)+'_n='+str(len(darks)), darkstack)
        
for ramp in ll_ramptimes:
    for i in range(len(zodi_cuts)-1):

        darks = get_dark_bcds(0, ll_gain_change_mjd, 2, zodi_cuts[i], zodi_cuts[i+1], ramp)
        if len(darks) > 0:
            darkstack = superdark(darks)
            np.save('./superdarks/LL_superdark_'+\
                    str(zodi_cuts[i])+'-'+str(zodi_cuts[i+1])+'_ramp='+str(ramp)+'_n='+str(len(darks)), darkstack)

        darks = get_dark_bcds(ll_gain_change_mjd, 55000, 2, zodi_cuts[i], zodi_cuts[i+1], ramp)
        if len(darks) > 0:
            darkstack = superdark(darks)
            np.save('./superdarks/LLa_superdark_'+\
                    str(zodi_cuts[i])+'-'+str(zodi_cuts[i+1])+'_ramp='+str(ramp)+'_n='+str(len(darks)), darkstack)