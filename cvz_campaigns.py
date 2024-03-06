from simla.database import query, bcd
import numpy as np
from tqdm import tqdm
import os

# This script will synthesize model zodi spectra at the CVZ 
# to make the model spectra that is subtracted from each model
# zodi spectrum during each campaign. (for LL)

# Load in the necessary data
campdata = np.load('./IRS_campaigns.npy')
camp_nums = campdata.T[0].astype(float)
camp_names = campdata.T[1]
camp_starts = campdata.T[2].astype(float)
camp_ends = campdata.T[3].astype(float)

# Query for info on CVZ observations
search = query(bcd.select(bcd.AORKEY, bcd.MJD_OBS)\
               .where(bcd.OBJECT=='NCVZ-dark-spot')\
               .where(bcd.CHNLNUM==2))
cvz_aor = search['AORKEY'].to_numpy()
cvz_time = search['MJD_OBS'].to_numpy()

if not os.path.exists('./cvz_campaign_spectra/'):
    os.mkdir('./cvz_campaign_spectra/')

for camp in tqdm(camp_nums):
    
    camp_index = np.where(camp_nums==camp)[0][0]
    
    # The decision to split the pre-C45 data into two
    # sections with two different window sizes was done
    # by-eye, but it makes the agreement much better.
    if camp < 23:
        
        window_start_index = camp_index - 5
        window_end_index = camp_index + 5
    
    elif 23 <= camp < 44:
    
        window_start_index = camp_index - 3
        window_end_index = camp_index + 3
        
        if window_end_index >= 45:
            window_end_index = 44
        
    # After the gain change in C45, they did something different.
    # This works okay.
    elif 44 <= camp:
        
        window_start_index = camp_index - 2
        window_end_index = camp_index
        
        if window_start_index < 45:
            window_start_index = 45

    if window_start_index < 0: window_start_index = 0

    # Average the spectra within the time window
    start_camp = camp_nums[window_start_index]
    end_camp = camp_nums[window_end_index]

    window_start_time = camp_starts[np.where(camp_nums==start_camp)][0]
    window_end_time = camp_ends[np.where(camp_nums==end_camp)][0]

    in_window_cvz_aors = cvz_aor[np.where((cvz_time>=window_start_time) & (cvz_time<=window_end_time))]

    cvz_zodis = [np.load('/home/work/simla/zodi_spectra/'+str(i)+'_ll_zodionly.npy')\
                 for i in in_window_cvz_aors]
    cvz_zodi = np.nanmean(cvz_zodis, axis=0)
    
    campname = camp_names[np.where(camp_nums==camp)][0]
    np.save('./cvz_campaign_spectra/cvz_for_'+campname, cvz_zodi)
    
# For SL, the average over all model CVZ spectra works well
sl_cvz_aors = query(bcd.select(bcd.AORKEY).where(bcd.OBJECT=='NCVZ-dark-spot')\
                    .where(bcd.CHNLNUM==0))['AORKEY'].to_numpy().tolist()
sl_cvz_aors = list(set(sl_cvz_aors)) # Unique AORKEYs only
sl_cvz_zodis = [np.load('/home/work/simla/zodi_spectra/'+str(aor)+'_sl_zodionly.npy') for aor in sl_cvz_aors]
sl_average_cvz = np.mean(np.asarray(sl_cvz_zodis), axis=0)
np.save('./cvz_campaign_spectra/cvz_average_sl_zodispec', sl_average_cvz)

############################################################
# Below this line is for plotting purposes only

from simla import tools
import matplotlib.pyplot as plt
from astropy.io import fits

search = query(bcd.select(bcd.FILE_NAME)\
               .where(bcd.OBJECT=='NCVZ-dark-spot')\
               .where(bcd.CHNLNUM==2))
cvz_bcds = search['FILE_NAME'].to_numpy()

extractor = tools.bcd_spectrum()

model_lam = np.load('/home/work/simla/zodi_spectra/ll_wavelengths.npy')

ts, f_avs = [], []
mod_fav = []
for cvzobs in tqdm(cvz_bcds):
    
    try:

        imdat = fits.getdata(cvzobs)
        imhead = fits.getheader(cvzobs)

        time = imhead['MJD_OBS']
        aor = imhead['AORKEY']
        campaign = imhead['CAMPAIGN']

        l, f = extractor.fullslit_bcd_spectrum(imdat, imhead, 1)

        zodi_spec = np.load('/home/work/simla/zodi_spectra/'+str(aor)+'_ll_zodionly.npy')
        cvz_spec = np.load('./cvz_campaign_spectra/cvz_for_'+campaign+'.npy')
        zodi_spec = zodi_spec - cvz_spec 

        mod_fav.append(np.nanmedian(zodi_spec))
        ts.append(time)
        f_avs.append(np.nanmedian(f))
        
    except: pass

sort_dat = sorted([[ts[i], f_avs[i], mod_fav[i]] for i in range(len(ts))])
pts = [i[0] for i in sort_dat]
pf_avs = [i[1] for i in sort_dat]
pmod = [i[2] for i in sort_dat]

px = [pts[i] for i in range(len(pts)) if -3<pf_avs[i]<4]
py = [pf_avs[i] for i in range(len(pf_avs)) if -3<pf_avs[i]<4]
plt.plot(pts, pmod, color='red')
plt.hexbin(x=px, y=py, cmap='rainbow', bins='log', mincnt=1, gridsize=70)
plt.ylabel('MJy/sr')
plt.xlabel('MJD')
plt.colorbar()
plt.savefig('./CVZ_campaign_plot.pdf', format='pdf')