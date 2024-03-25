from simla.database import query, bcd, qFull, \
    wmap, shardphot, qShard, pq, shardpos, judge2, zodi, galcoords
import numpy as np
from simla import tools
from astropy.io import fits
from tqdm import tqdm
import numpy.ma as ma
import matplotlib.pyplot as plt
import glob
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy
from matplotlib.pyplot import figure
import os
import pypika
import time
from idlpy import IDL
from scipy import ndimage
from io_correct import SL_IO_correct_image
from scipy import stats

def simlacube(inputs):

    simlapath = os.path.dirname(os.path.realpath(__file__))
    cube_start_time = time.time()
    
    IDL.run('.RESET_SESSION')
    
    # get a representative time for the cube
    cube_t = np.mean(query(qFull.select(bcd.MJD_OBS)\
              .where(bcd.AORKEY==inputs['cube_AORKEY']))['MJD_OBS'])

    exposure_stats = query(qFull.select(bcd.SAMPTIME, bcd.RAMPTIME, bcd.EXPTOT_T)\
              .where(bcd.CHNLNUM==inputs['channel'])\
              .where(bcd.AORKEY==inputs['cube_AORKEY']))
    ramp, samp, expt = exposure_stats['RAMPTIME'][0], exposure_stats['SAMPTIME'][0], \
        exposure_stats['EXPTOT_T'][0]
        
    map_stats = query(qFull.select(bcd.STEPSPAR, bcd.STEPSPER, bcd.NCYCLES)\
              .where(bcd.CHNLNUM==inputs['channel'])\
              .where(bcd.AORKEY==inputs['cube_AORKEY']))
    per, par = map_stats['STEPSPER'][0], map_stats['STEPSPAR'][0]
    
    # get subslits approved by Judge1 and Judge2 within the delta t
    # SAME RAMPTIME IS VERY IMPORTANT
    darks = query(qShard.select(bcd.FILE_NAME, bcd.DCEID, bcd.AORKEY, bcd.MJD_OBS, 
                                zodi.LL1_C,
                                judge2.SUBORDER, judge2.SUBSLIT)\
              .where(bcd.CHNLNUM==inputs['channel'])\
              .where((bcd.RAMPTIME>ramp-0.01)&(bcd.RAMPTIME<ramp+0.01))\
              .where(bcd.OBJECT!='NCVZ-dark-spot')\
              .where((bcd.MJD_OBS>=cube_t-inputs['delta_t'])&(bcd.MJD_OBS<=cube_t+inputs['delta_t']))\
              .where((shardphot.BACKSUB_PHOT<=inputs['J1_cut'])&(shardphot.BACKSUB_PHOT>=-inputs['J1_cut']))\
              .where((shardphot.BACKSUB_PHOT<-0.001)|(shardphot.BACKSUB_PHOT>0.001))\
              .where((judge2.F_MEDIAN<=inputs['J2_cut'])&(judge2.F_MEDIAN)>=-inputs['J2_cut']))

    # parse database output
    fnames = darks['FILE_NAME'].to_numpy()
    dceids = darks['DCEID'].to_numpy().astype(int)
    aorkeys = darks['AORKEY'].to_numpy().astype(int)
    mjdobs = darks['MJD_OBS'].to_numpy()
    suborders = darks['SUBORDER'].to_numpy().astype(int)
    subslits = darks['SUBSLIT'].to_numpy().astype(int)
    zodis = darks['LL1_C'].to_numpy()
    
    # if desired, mask out in-AOR bcds
    if inputs['ignore_in_AORKEY']:
        aor_mask = np.where(aorkeys!=inputs['cube_AORKEY'])
        fnames = fnames[aor_mask]
        dceids = dceids[aor_mask]
        aorkeys = aorkeys[aor_mask]
        mjdobs = mjdobs[aor_mask]
        suborders = suborders[aor_mask]
        subslits = subslits[aor_mask]
        zodis = zodis[aor_mask]
    
    # mask the results of the data base output based on the zodi cut
    if inputs['zodi_cut'] is not None:
        cube_zodi = query(zodi.select(zodi.LL1_C).where(zodi.AORKEY==inputs['cube_AORKEY']))['LL1_C'][0]
        zodi_diff_tolerance = inputs['zodi_cut'] # plus/minus
        zodi_mask = np.where((zodis>cube_zodi-zodi_diff_tolerance)&(zodis<cube_zodi+zodi_diff_tolerance))
        # zodi_mask = np.where((zodis<cube_zodi-zodi_diff_tolerance)|(zodis>cube_zodi+zodi_diff_tolerance))
        fnames = fnames[zodi_mask]
        dceids = dceids[zodi_mask]
        aorkeys = aorkeys[zodi_mask]
        mjdobs = mjdobs[zodi_mask]
        suborders = suborders[zodi_mask]
        subslits = subslits[zodi_mask]
    else: 
        inputs['zodi_cut'] = 1000
        cube_zodi = 0
        zodi_diff_tolerance = 1000
    
    chnlnum = inputs['channel']
    ordername = ['SL', 'SH', 'LL', 'LH'][chnlnum]

    # load in files needed for IO removal
    if chnlnum == 0 and inputs['io_correct'] == True:
        flatfield = fits.getdata(simlapath+'/calib/b0_flatfield.fits')[0]
        SL1_mask = fits.getdata('/home/work/simla/calib/masks/irs_mask_SL1.fits')
        SL2_mask = fits.getdata('/home/work/simla/calib/masks/irs_mask_SL2.fits')
        SL1_mask = np.where(SL1_mask==1, 1, np.nan)
        SL2_mask = np.where(SL2_mask==1, 1, np.nan)
        io_files = {'flatfield': flatfield,
                    'SL1_mask': SL1_mask,
                    'SL2_mask': SL2_mask}
        
    # making sure the IO signal makes it into the background
    allmask = fits.getdata('/home/work/simla/calib/all_masks/irs_mask_'+ordername+'_ALL.fits')
    offorder_mask = np.where(allmask==0, 1, 0)
    
    # load in subslit masks
    subslit_masks = [
        [[fits.getdata(i) for i in sorted(glob.glob('/home/work/simla/calib/subslit_masks_4per_trim_6_subslits/*SL1*'))], 
         [fits.getdata(i) for i in sorted(glob.glob('/home/work/simla/calib/subslit_masks_4per_trim_6_subslits/*SL2*'))]],
        None,
        [[fits.getdata(i) for i in sorted(glob.glob('/home/work/simla/calib/subslit_masks_4per_trim_6_subslits/*LL1*'))], 
         [fits.getdata(i) for i in sorted(glob.glob('/home/work/simla/calib/subslit_masks_4per_trim_6_subslits/*LL2*'))]],
        None
    ]
    
    # open lists that will be returned
    dark_stack = []
    times = []
    unc_stack = []
    depth_stack = []
    
    # loop through the unique AORKEYs
    unique_aorkeys = np.unique(aorkeys)
    for aorkey in unique_aorkeys:

        # get the unique DCEIDS in this AOR
        aor_indices = np.where(aorkeys==aorkey)
        unique_dceids_in_aor = np.unique(dceids[aor_indices])

        # load in the frames that will be subtracted off of everything in this AOR
        zodi_image = np.load(simlapath+'/zodi_images/'+str(aorkey)+'_'+ordername+'.npy')
        
        # get interpolated superdark for this AOR
        if inputs['superdark']:
            superdark = np.load(simlapath+'/tailored_superdarks/'+str(aorkey)+'_'+ordername+'.npy')

        # loop through the DCEIDs in this AOR
        for dceid in unique_dceids_in_aor:

            indices = np.where(dceids==dceid)
            fname = fnames[indices][0]

            # load in the BCD
            with fits.open(fname) as hdul:
                image_header = hdul[0].header
                image_data = hdul[0].data

            # get the image data in zero-zodi space
            subtracted_image = image_data - zodi_image
            
            # subtract the superdark
            if inputs['superdark']:
                subtracted_image = subtracted_image - superdark

            # prepare the dark mask, a mask of the dark subslits in this DCEID
            dark_mask = np.zeros((128, 128))

            # loop only through the Judge selected subslits
            for i in indices[0]:
                
                suborder_num = suborders[i]
                subslit_num = subslits[i]
                
                # add the subslit to the mask
                mask = subslit_masks[chnlnum][suborder_num-1][subslit_num]
                dark_mask += mask
            
            # this makes sure that the IO is included
            dark_mask += offorder_mask

            # apply the selected subslit masks to the BCD and add it to the stack
            dark_mask = np.where(dark_mask>1, 1, dark_mask) # deal with overlaps
            extracted_dark = np.where(dark_mask==1, subtracted_image, np.nan)
            
            dark_stack.append(extracted_dark)
            depth_stack.append(dark_mask)

            times.append(mjdobs[indices][0])
            
            # load in the uncertainty BCD, extract the same subslits, add to a stack
            unc_image = fits.getdata((fname.replace('bcd.','func.')))
            extracted_unc = np.where(dark_mask==1, unc_image, 0)
            unc_stack.append(extracted_unc)

    dark_stack = np.asarray(dark_stack)
    unc_stack = np.asarray(unc_stack)

    # create weights for the dark using the times
    times = np.asarray(times)
    # weights = 1/(1+np.abs(cube_t-times))
    weights = np.ones_like(times)

    # function to iteratively trim the stack comparing pixels to others in the stack
    def iterative_trim(stack, unc_stack, iters=1, stdev=3):
        for i in range(iters):
            std_dev_im = np.nanstd(stack, axis=0)
            mean_im = np.nanmean(stack, axis=0)
            unc_stack = np.where( (stack>mean_im+(stdev*std_dev_im)) | (stack<mean_im-(stdev*std_dev_im)), np.nan, unc_stack)
            stack = np.where( (stack>mean_im+(stdev*std_dev_im)) | (stack<mean_im-(stdev*std_dev_im)), np.nan, stack)
        return stack, unc_stack

    # apply trimming and average
    trimmed_dark_stack, trimmed_unc_stack = iterative_trim(dark_stack, unc_stack, iters=3, stdev=10)
    # trimmed_dark_stack = np.ma.MaskedArray(trimmed_dark_stack, mask=np.isnan(trimmed_dark_stack))
    # trimmed_dark_stack = np.ma.average(trimmed_dark_stack, axis=0, weights=weights)
    # trimmed_dark_stack = np.ma.median(trimmed_dark_stack, axis=0)
    trimmed_dark_stack = np.nanmean(trimmed_dark_stack, axis=0)

    # load in zodi and leftover images for the cube AOR
    zodi_image = np.load(simlapath+'/zodi_images/'+str(inputs['cube_AORKEY'])+'_'+ordername+'.npy')
    
    # add the zodi and the leftover in to the dark
    dark = trimmed_dark_stack + zodi_image
    
    # # create the dark uncertainty image
    def stack_uncs(unc_images, weights=None):
        '''
        This function takes a list of 2D image arrays where pixel values
        are uncertainties and returns one 2D image array that is the
        weighted average.

        This method is translated from the IDL code that performs the
        same function for averaging background images in the CUBISM program.

        Parameters:
        unc_images (list): The list of uncertainty images to average
        weights (list, optional): If provided, will weight each uncertainty image as unc_image_array*weighting_array. 
        Must be same length as unc_image_array.
        Returns:
        array: Average uncertainty image.
        '''
        # first there's a toggle to zero out max and min,
        # seems like a good idea to me so I'll hard code that
        # this zeros the max pixel of whichever image has the
        # most maximum max value
        unc_images = np.array(unc_images)
        unc_images[np.unravel_index(np.argmax(unc_images), unc_images.shape)] = 0
        unc_images[np.unravel_index(np.argmin(unc_images), unc_images.shape)] = 0

        if weights: unc_images *= weights

        return np.sqrt(np.nansum(unc_images**2, axis=0))/unc_images.shape[0]
    dark_unc = stack_uncs(trimmed_unc_stack, weights=None)
    
    # the depth image will show the number of subslits that contributed
    depth_image = np.nansum(depth_stack, axis=0)
    depth_mask = fits.getdata('/home/work/simla/calib/masks_4per_trim/irs_mask_'+ordername+'1.fits')+\
                 fits.getdata('/home/work/simla/calib/masks_4per_trim/irs_mask_'+ordername+'2.fits')
    depth_image = depth_image * np.where(depth_mask==0, np.nan, depth_mask)
    
    # gather bcds that will be in the cube
    cube_bcds = query(bcd.select(bcd.FILE_NAME).where(bcd.AORKEY==inputs['cube_AORKEY'])\
                     .where(bcd.CHNLNUM==inputs['channel']))['FILE_NAME'].tolist()
    
    # create an interpolated superdark for the cube AOR, and add it to the background
    if inputs['superdark']:
        cube_superdark = np.load(simlapath+'/tailored_superdarks/'+\
                          str(inputs['cube_AORKEY'])+'_'+ordername+'.npy')
        dark = dark + cube_superdark
    
    # if SL, create IO image from each cube BCD and remove
    temp_dir = simlapath+'/temp/'
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
        np.save(temp_dir+'SPITZER_DUMMY', [])
    if inputs['channel'] == 0 and inputs['io_correct'] == True:
        
        os.system('rm '+temp_dir+'SPITZER*')
        os.system('cp '+cube_bcds[0].split('SPITZER')[0]+'* '+temp_dir)
        cube_bcds = glob.glob(temp_dir+'*bcd.fits')
        for b in cube_bcds:
            imdat = fits.getdata(b)
            darksub_imdat = imdat - dark
            io_correct_im = SL_IO_correct_image(darksub_imdat, io_files)
            fixed_im = imdat - io_correct_im
            hdu = fits.PrimaryHDU(fixed_im)
            hdulist = fits.HDUList([hdu])
            hdulist[0].header = fits.getheader(b)
            hdulist.writeto(b, overwrite=True)
            hdulist.close()
        cube_bcds = glob.glob(temp_dir+'*bcd.fits')
        
    header = fits.getheader(cube_bcds[0])
        
    # make the cubes
    tools.build_cube(cube_bcds, inputs['channel'], 1, dark, 
        dark_unc, inputs['savename']+'_'+ordername+'1.fits', autobp=True)
    tools.build_cube(cube_bcds, inputs['channel'], 2, dark, 
        dark_unc, inputs['savename']+'_'+ordername+'2.fits', autobp=True)
    
    cube_end_time = time.time()
    
    # Remove the "_unc" from the .cpj files that appears for some reason
    os.system('mv '+inputs['savename']+'_'+ordername+'1_unc.cpj'+' '+inputs['savename']+'_'+ordername+'1.cpj')
    os.system('mv '+inputs['savename']+'_'+ordername+'2_unc.cpj'+' '+inputs['savename']+'_'+ordername+'2.cpj')
    
    # Save the generated background and depth map
    np.save(inputs['savename']+'_'+ordername+'_background', dark)
    np.save(inputs['savename']+'_'+ordername+'_background_unc', dark_unc)
    np.save(inputs['savename']+'_'+ordername+'_background_depth', depth_image)
    
    # Save the auto-generated bad pixel list
    IDL.cpjpath1 = inputs['savename']+'_'+ordername+'1.cpj'
    IDL.cpjpath2 = inputs['savename']+'_'+ordername+'2.cpj'
    IDL.run('.run /usr/local/idl/idl_library/cubism/cubism/cube/cubeproj_load.pro')
    IDL.run('cube1=cubeproj_load(cpjpath1)')
    IDL.run('cube2=cubeproj_load(cpjpath2)')
    IDL.run('cube1->SaveBadPixels, "'+inputs['savename']+'_'+ordername+'1.bpl"')
    IDL.run('cube2->SaveBadPixels, "'+inputs['savename']+'_'+ordername+'2.bpl"')
    
    # Save the shards used as backgrounds
    shardlistfile = open(inputs['savename']+'_'+ordername+'_shardlist.txt', 'w')
    shardlistfile.write('AORKEY    DCEID    ORDER1_SHARDS    ORDER2_SHARDS')
    shardlistfile.write('\n')
    for dceid in np.unique(dceids):
        indices = np.where(dceids==dceid)
        aorkey = aorkeys[indices][0]
        order1_shards = []
        order2_shards = []
        for i in indices[0]: 
            suborder_num = suborders[i]
            if suborder_num == 1:
                order1_shards.append(subslits[i])
            elif suborder_num == 2:
                order2_shards.append(subslits[i])
        shardlistfile.write(str(aorkey)+'    '+str(dceid)+'    '+str(order1_shards)+'    '+str(order2_shards))
        shardlistfile.write('\n')
        
    # if wanted, load in the new cubes and do diagnostics on them
    if inputs['dev']:
        
        if inputs['aperture'] == 'standard':
            head = header
            ra, dec = head['RA_REF'], head['DEC_REF']
            ap = tools.ellipse_region({'ra':ra,'dec':dec,'inc':0,'pa':0,'distance':0}, 
                                   inner_radius=20, angular=True, project=False).coords
            inputs['aperture'] = ap
        
        cube1 = tools.load_cube(inputs['savename']+'_'+ordername+'1.fits')
        cube2 = tools.load_cube(inputs['savename']+'_'+ordername+'2.fits')
        
        l1, f1 = tools.spectral_extraction(inputs['aperture'], cube1)
        l2, f2 = tools.spectral_extraction(inputs['aperture'], cube2)
        
        del cube1 # no longer need to have them loaded in, and having them may
        del cube2 # cause problems when building many cubes
        
        def get_rms(ordername, lam, spec):

            cont = {
                'SL': [[5.2, 6.0], [6.5, 7.1], [9.0, 10.3], [13.0, 14.0]],
                'LL': [[15.0, 16.2], [18.9, 20.0], [20.0, 22.5], [22.5, 25.0], [29.0, 31.0], [31.0, 33.0]]
            }[ordername]

            sub_continuua = []
            for c in cont:

                # get anchor points for this continuum section
                lower, upper = c[0], c[1]
                contmask = np.where((lam>lower)&(lam<upper))
                cont_l, cont_s = lam[contmask], spec[contmask]
                anchor_low = np.nanmedian(cont_s[:3])
                anchor_high = np.nanmedian(cont_s[-3:])
                low_l, high_l = np.nanmedian(cont_l[:3]), np.nanmedian(cont_l[-3:])

                # create continuum line and subtract
                m = (anchor_high - anchor_low)/(high_l - low_l)
                b = -m*low_l+anchor_low
                cont = m*cont_l + b
                sub_spec = cont_s - cont
                sub_continuua.extend(sub_spec)

            # get RMS
            rms = np.sqrt(np.nanmean(np.asarray(sub_continuua)**2))
            
            # get MAD
            mad = stats.median_abs_deviation(sub_continuua)

            return rms, mad

        cube1_rms, cube1_mad = get_rms(ordername, l1, f1) 
        cube2_rms, cube2_mad = get_rms(ordername, l2, f2)
        
        # if there is a cooresponding gold cube, compare
        if inputs['isgold']:
            
            # load golds and extract spectra
            gold_1 = tools.load_cube(glob.glob(inputs['gold_cube']+'/*'+ordername+'1.fits')[0])
            gold_2 = tools.load_cube(glob.glob(inputs['gold_cube']+'/*'+ordername+'2.fits')[0])
            
            gl1, gf1 = tools.spectral_extraction(inputs['aperture'], gold_1)
            gl2, gf2 = tools.spectral_extraction(inputs['aperture'], gold_2)
            
            gold1_rms, gold1_mad = get_rms(ordername, gl1, gf1)
            gold2_rms, gold2_mad = get_rms(ordername, gl2, gf2)
            
            # residuals
            res_1, res_2 = f1 - gf1, f2 - gf2
            res_1_rms = np.sqrt(np.nanmean(np.asarray(res_1)**2))
            res_2_rms = np.sqrt(np.nanmean(np.asarray(res_2)**2))
    
    class Diagnostics:
        def __init__(self):
            
            cube_bcds = query(bcd.select(bcd.FILE_NAME).where(bcd.AORKEY==inputs['cube_AORKEY'])\
                 .where(bcd.CHNLNUM==inputs['channel']))['FILE_NAME'].tolist()
            
            self.background = dark
            self.background_unc = dark_unc
            self.depth_map = depth_image
            self.example_header = fits.getheader(cube_bcds[0])
            self.RAMPTIME = ramp
            self.SAMPTIME = samp
            self.EXPTOT_T = expt
            self.inputs = inputs
            
            slt_info = {
            'sl1': {
                'width': 3.7 / 3600,
                'length': 57 / 3600, 
            },
            'sl2': {
                'width': 3.6 / 3600,
                'length': 57 / 3600,
            },
            'll1': {
                'width': 10.7 / 3600,
                'length': 168 / 3600,
            },
            'll2': {
                'width': 10.5 / 3600,
                'length': 168 / 3600,
            },
            }
            
            little_name = ['sl', None, 'll'][inputs['channel']]
            o1_w, o1_l = slt_info[little_name+'1']['width'], slt_info[little_name+'1']['length']
            depth_indicator1 = ramp*map_stats['NCYCLES'][0]*(o1_w/per)*(o1_l/par)
            o2_w, o2_l = slt_info[little_name+'2']['width'], slt_info[little_name+'2']['length']
            depth_indicator2 = ramp*map_stats['NCYCLES'][0]*(o2_w/per)*(o2_l/par)
            
            self.o1_depth = depth_indicator1
            self.o2_depth = depth_indicator2
            
            if inputs['dev']:
                
                self.spec1 = l1, f1
                self.spec2 = l2, f2
                self.cube_rms = cube1_rms, cube2_rms
                self.cube_mad = cube1_mad, cube2_mad
                
                # Stats
                self.LL1_C_zodi = query(zodi.select(zodi.LL1_C).where(zodi.AORKEY==inputs['cube_AORKEY']))['LL1_C'][0]
                self.cube_time = cube_end_time - cube_start_time
                self.bcds_in_AOR = len(cube_bcds)
                self.AORs_in_background = len(unique_aorkeys)
                self.AOR_MJD = cube_t
                header = fits.getheader(cube_bcds[0])
                self.object_name = header['OBJECT']
                galaxy_coordinates_search = query(galcoords.select(galcoords.GALACTIC_L, galcoords.GALACTIC_B)\
                                                  .where(galcoords.DCEID==header['DCEID']))
                self.gal_coords = (galaxy_coordinates_search['GALACTIC_L'][0], galaxy_coordinates_search['GALACTIC_B'][0])
                #
                
                # For the zodi histogram
                self.zodis = zodis
                self.zodi_range = (cube_zodi-zodi_diff_tolerance, cube_zodi+zodi_diff_tolerance)
                #
                
                ### Flags ###

                # - Are SL2 + SL1 spectra continuous?
                allowed_difference = 20 # %
                o1 = np.nanmedian(f1[0:3])
                o2 = np.nanmedian(f2[-3:])
                if (np.abs(o1-o2)/((o1+o2)/2))*100 > allowed_difference:
                    self.spectral_continuity = False
                else: self.spectral_continuity = True

                # - Is the depth map sufficiently deep?
                minimum_depth = 50 # per shard
                # If *any* shard is not this deep, the flag is thrown
                if np.nansum(np.where(depth_image<minimum_depth)) > 0:
                    self.sufficient_depth = False
                else: self.sufficient_depth = True
                
                # See whether any selected shards are from this AOR
                # (done for both suborders)
                order1_in_aor = len(aorkeys[np.where((aorkeys==int(inputs['cube_AORKEY']))\
                                                           &(suborders==1))])
                order2_in_aor = len(aorkeys[np.where((aorkeys==int(inputs['cube_AORKEY']))\
                                                           &(suborders==2))])
                self.order1_AOR_shards = order1_in_aor
                self.order2_AOR_shards = order2_in_aor
                
                # How many selected shards are *not* in this AOR?
                order1_not_in_aor = len(aorkeys[np.where((aorkeys!=int(inputs['cube_AORKEY']))\
                                                               &(suborders==1))])
                order2_not_in_aor = len(aorkeys[np.where((aorkeys!=int(inputs['cube_AORKEY']))\
                                                               &(suborders==2))])
                self.order1_nonAOR_shards = order1_not_in_aor
                self.order2_nonAOR_shards = order2_not_in_aor
                #############

                
                if inputs['isgold']:
                    self.gold_spec1 = gl1, gf1
                    self.gold_spec2 = gl2, gf2
                    self.gold_rms = gold1_rms, gold2_rms
                    self.gold_mad = gold1_mad, gold2_mad
                    self.resuldual_spectra = res_1, res_2
                    self.residual_rms = res_1_rms, res_2_rms
                    
        def generate_ds9_regionfiles(self):

            rects = query(qShard.select(shardpos.C0_R, shardpos.C1_R, shardpos.C2_R, shardpos.C3_R, 
                                        shardpos.C0_D, shardpos.C1_D, shardpos.C2_D, shardpos.C3_D,
                                        shardphot.BACKSUB_PHOT, judge2.F_MEDIAN, bcd.AORKEY)\
              .where(bcd.CHNLNUM==self.inputs['channel'])\
              .where((bcd.RAMPTIME>ramp-0.01)&(bcd.RAMPTIME<ramp+0.01))\
              .where(bcd.OBJECT!='NCVZ-dark-spot')\
              .where((shardphot.BACKSUB_PHOT<-0.001)|(shardphot.BACKSUB_PHOT>0.001))\
              .where((bcd.MJD_OBS>=cube_t-inputs['delta_t'])&(bcd.MJD_OBS<=cube_t+inputs['delta_t'])))
            
            unique_aors = np.unique(rects['AORKEY'])
            wise_lookup = {}
            for aor in unique_aors:
                wise_file = tools.unwise_to_irsawise(query(pq.from_(wmap).join(bcd).using("DCEID")\
                                .select(wmap.SLIT1_WISE_FILE).where(bcd.AORKEY==inputs['cube_AORKEY']))\
                                                   ['SLIT1_WISE_FILE'][0])
                wise_lookup[str(aor)] = wise_file
                
            corners_lookup = {}
            for aor in unique_aors:
                corners_lookup[str(aor)] = []
            
            for r in range(len(rects['C0_R'].to_numpy())):
                
                corners = [[rects['C0_R'][r], rects['C0_D'][r]],
                           [rects['C1_R'][r], rects['C1_D'][r]],
                           [rects['C2_R'][r], rects['C2_D'][r]],
                           [rects['C3_R'][r], rects['C3_D'][r]]]
                
                if rects['BACKSUB_PHOT'][r] <- self.inputs['J1_cut'] or rects['BACKSUB_PHOT'][r] > self.inputs['J1_cut']\
               and rects['F_MEDIAN'][r] <- self.inputs['J2_cut'] or rects['F_MEDIAN'][r] > self.inputs['J2_cut']:
                    color = 'red'
                        
                elif rects['BACKSUB_PHOT'][r] >- self.inputs['J1_cut'] and rects['BACKSUB_PHOT'][r] < self.inputs['J1_cut']\
                 and rects['F_MEDIAN'][r] <- self.inputs['J2_cut'] or rects['F_MEDIAN'][r] > self.inputs['J2_cut']:
                    color = 'blue'
                    
                elif rects['BACKSUB_PHOT'][r] <- self.inputs['J1_cut'] or rects['BACKSUB_PHOT'][r] > self.inputs['J1_cut']\
                 and rects['F_MEDIAN'][r] >- self.inputs['J2_cut'] and rects['F_MEDIAN'][r] < self.inputs['J2_cut']:
                    color = 'yellow'
                    
                elif rects['BACKSUB_PHOT'][r] >- self.inputs['J1_cut'] and rects['BACKSUB_PHOT'][r] < self.inputs['J1_cut']\
                 and rects['F_MEDIAN'][r] >- self.inputs['J2_cut'] and rects['F_MEDIAN'][r] < self.inputs['J2_cut']:
                    color = 'green'
                
                aor = rects['AORKEY'][r]
                
                corners_lookup[str(aor)].append([corners, color])
            
            chnlnum = self.inputs['channel']
            ordername = ['SL', 'SH', 'LL', 'LH'][chnlnum]
            region_filename = self.inputs['savename']+'_'+ordername+'_shardregions.reg'
            regionfile = open(region_filename, 'w')
                
            for aor in unique_aors:
                
                wisefile = wise_lookup[str(aor)]
                regionfile.write('# AORKEY: '+str(aor))
                regionfile.write('\n')
                regionfile.write('# WISE_FILE: '+wisefile)
                regionfile.write('\n')
                
                data = corners_lookup[str(aor)]
                for i in range(len(data)):
                    
                    corners = data[i][0]
                    color = data[i][1]

                    line = 'fk5; polygon('
                    for c in range(len(corners)):
                        if c != len(corners)-1:
                            line += str(corners[c][0])+','
                            line += str(corners[c][1])+','
                        else:
                            line += str(corners[c][0])+','
                            line += str(corners[c][1])
                    line += ')'
                    line += ' # color='+color

                    regionfile.write(line)
                    regionfile.write('\n')
                regionfile.write('\n')
            
        def show_background(self, vmin=0, vmax=50, show=True):
            plt.imshow(dark, origin='lower', interpolation='none', vmin=vmin, vmax=vmax)
            plt.title('Background')
            plt.colorbar()
            if show:
                plt.show()
            else:
                return plt
            
        def show_background_unc(self, vmin=0, vmax=5, show=True):
            plt.imshow(dark_unc, origin='lower', interpolation='none', vmin=vmin, vmax=vmax)
            plt.title('Background Uncertainty')
            plt.colorbar()
            if show:
                plt.show()
            else:
                return plt
            
        def show_depth_map(self, show=True):
            plt.imshow(depth_image, origin='lower', interpolation='none')
            plt.title('Depth Map')
            plt.colorbar()
            if show:
                plt.show()
            else:
                return plt
            
        def show_weights(self, show=True):
            d = sorted([[times[i], weights[i]] for i in range(len(weights))])
            t = [i[0] for i in d]
            w = [i[1] for i in d]
            plt.scatter(t, w)
            plt.title('Weights')
            plt.plot(t, w)
            plt.axvline(cube_t, color='red')
            plt.xlabel('MJD')
            plt.ylabel('Weight')
            if show:
                plt.show()
            else:
                return plt
            
        def show_dark_spectra(self, show=True):
            
            dark_spec1 = query(qShard.select(judge2.SPEC0, judge2.SPEC1, judge2.SPEC2,
                                             judge2.SPEC3, judge2.SPEC4, judge2.SPEC5,
                                             judge2.SPEC6, judge2.SPEC7, judge2.SPEC8, judge2.SPEC9)\
           .where(judge2.CHNLNUM==inputs['channel']).where(judge2.SUBORDER==1)\
           .where((bcd.MJD_OBS>=cube_t-inputs['delta_t'])&(bcd.MJD_OBS<=cube_t+inputs['delta_t']))\
           .where((shardphot.BACKSUB_PHOT<=inputs['J1_cut'])&(shardphot.BACKSUB_PHOT>=-inputs['J1_cut']))\
           .where((judge2.F_MEDIAN<=inputs['J2_cut'])&(judge2.F_MEDIAN)>=-inputs['J2_cut']))
            
            dark_spec2 = query(qShard.select(judge2.SPEC0, judge2.SPEC1, judge2.SPEC2,
                                             judge2.SPEC3, judge2.SPEC4, judge2.SPEC5,
                                             judge2.SPEC6, judge2.SPEC7, judge2.SPEC8, judge2.SPEC9)\
           .where(judge2.CHNLNUM==inputs['channel']).where(judge2.SUBORDER==2)\
           .where((bcd.MJD_OBS>=cube_t-inputs['delta_t'])&(bcd.MJD_OBS<=cube_t+inputs['delta_t']))\
           .where((shardphot.BACKSUB_PHOT<=inputs['J1_cut'])&(shardphot.BACKSUB_PHOT>=-inputs['J1_cut']))\
           .where((judge2.F_MEDIAN<=inputs['J2_cut'])&(judge2.F_MEDIAN)>=-inputs['J2_cut']))
            
            spectra_1 = np.asarray([dark_spec1['SPEC'+str(i)] for i in range(10)]).T
            spectra_2 = np.asarray([dark_spec2['SPEC'+str(i)] for i in range(10)]).T
            
            x, y = [], []
            for i in spectra_1:
                for j in range(10):
                    if -5<i[j]<5:
                        x.append(j)
                        y.append(i[j])
            for i in spectra_2:
                for j in range(10):
                    if -5<i[j]<5:
                        x.append(j+10)
                        y.append(i[j])
            plt.hexbin(x=x, y=y, cmap='rainbow', bins='log', mincnt=1, gridsize=100)
            plt.title('Spectra of Selected Darks')
            plt.xlabel('Bin Number')
            plt.ylabel('Binned Surface Brightness [MJy/sr]')
            plt.colorbar()
            if show:
                plt.show()
            else:
                return plt
            
        def spectrum_of_background(self, show=False):
            
            extractor = tools.bcd_spectrum()
            l1, f1 = extractor.fullslit_bcd_spectrum(self.background, self.example_header, 1)
            l2, f2 = extractor.fullslit_bcd_spectrum(self.background, self.example_header, 2)
            
            if inputs['isgold'] == True:
                o1_gold_bg = fits.getdata(glob.glob(inputs['gold_cube']+'/*'+ordername+'1_gold_dark.fits')[0])
                o2_gold_bg = fits.getdata(glob.glob(inputs['gold_cube']+'/*'+ordername+'2_gold_dark.fits')[0])
                l1, gf1 = extractor.fullslit_bcd_spectrum(o1_gold_bg, self.example_header, 1)
                l2, gf2 = extractor.fullslit_bcd_spectrum(o2_gold_bg, self.example_header, 2)
                plt.plot(l2, gf2, color='gold')
                plt.plot(l1, gf1, color='gold')
            
            plt.scatter(l1, f1, alpha=0.5)
            plt.title('Spectrum of Background')
            plt.plot(l1, f1)
            plt.scatter(l2, f2, alpha=0.5)
            plt.plot(l2, f2)
            plt.xlabel('Wavelength [micron]')
            plt.ylabel('Surface Brightness [MJy/sr]')
            # plt.yscale('log')
            if show:
                plt.show()
            else:
                return plt
            
            
        def show_extracted_spectra(self, show=True):
            plt.scatter(l1, f1, alpha=0.5)
            plt.title('Spectrum of Cubes')
            plt.plot(l1, f1)
            plt.scatter(l2, f2, alpha=0.5)
            plt.plot(l2, f2)
            plt.xlabel('Wavelength [micron]')
            plt.ylabel('Surface Brightness [MJy/sr]')
            if show:
                plt.show()
            else:
                return plt
            
        def make_stoplight(self, suborder, show=True):
            
            from astropy.visualization import (MinMaxInterval, AsinhStretch,
                                               ImageNormalize)

            irsawise_file = tools.unwise_to_irsawise(query(pq.from_(wmap).join(bcd).using("DCEID")\
                            .select(wmap.SLIT1_WISE_FILE).where(bcd.AORKEY==inputs['cube_AORKEY']))\
                                               ['SLIT1_WISE_FILE'][0])

            def zoom_image(zoom_center, zoom_size, imagefile):

                image_data = fits.open(imagefile)[0].data
                image_header = fits.open(imagefile)[0].header
                wcs = WCS(image_header)

                zoom_center = SkyCoord(zoom_center[0], zoom_center[1], unit='deg')
                crop_fits = Cutout2D(image_data, astropy.wcs.utils.skycoord_to_pixel(zoom_center,wcs), zoom_size, wcs=wcs)
                wcs = crop_fits.wcs
                image_data = crop_fits.data
                image_header.update(crop_fits.wcs.to_header()) 

                return image_data, image_header
            
            all_corners = []

            rects = query(qShard.select(shardpos.C0_R, shardpos.C1_R, shardpos.C2_R, shardpos.C3_R, 
                                        shardpos.C0_D, shardpos.C1_D, shardpos.C2_D, shardpos.C3_D,
                                        shardphot.BACKSUB_PHOT, judge2.F_MEDIAN)\
              .where((bcd.CHNLNUM==inputs['channel'])&(bcd.AORKEY==inputs['cube_AORKEY']))\
              .where(judge2.SUBORDER==suborder))
            
            for r in range(len(rects['C0_R'].to_numpy())):
                corners = [[rects['C0_R'][r], rects['C0_D'][r]],
                           [rects['C1_R'][r], rects['C1_D'][r]],
                           [rects['C2_R'][r], rects['C2_D'][r]],
                           [rects['C3_R'][r], rects['C3_D'][r]]]
                
                if rects['BACKSUB_PHOT'][r] <- inputs['J1_cut'] or rects['BACKSUB_PHOT'][r] > inputs['J1_cut']\
               and rects['F_MEDIAN'][r] <- inputs['J2_cut'] or rects['F_MEDIAN'][r] > inputs['J2_cut']:
                    color = 'red'
                        
                elif rects['BACKSUB_PHOT'][r] >- inputs['J1_cut'] and rects['BACKSUB_PHOT'][r] < inputs['J1_cut']\
                 and rects['F_MEDIAN'][r] <- inputs['J2_cut'] or rects['F_MEDIAN'][r] > inputs['J2_cut']:
                    color = 'blue'
                    
                elif rects['BACKSUB_PHOT'][r] <- inputs['J1_cut'] or rects['BACKSUB_PHOT'][r] > inputs['J1_cut']\
                 and rects['F_MEDIAN'][r] >- inputs['J2_cut'] and rects['F_MEDIAN'][r] < inputs['J2_cut']:
                    color = 'yellow'
                    
                elif rects['BACKSUB_PHOT'][r] >- inputs['J1_cut'] and rects['BACKSUB_PHOT'][r] < inputs['J1_cut']\
                 and rects['F_MEDIAN'][r] >- inputs['J2_cut'] and rects['F_MEDIAN'][r] < inputs['J2_cut']:
                    color = 'green'
                    
                all_corners.append([corners, color])
            
            j1_passed, j2_passed = 0, 0
            green_num, non_red_num = 0, 0
            for i in range(len(all_corners)):
                color = all_corners[i][1]
                if color == 'green' or color == 'blue': j1_passed += 1
                elif color == 'green' or color == 'yellow': j2_passed += 1
                if color == 'green': green_num += 1
                if color != 'red': non_red_num += 1
            j1_fraction = j1_passed / len(all_corners)
            j2_fraction = j2_passed / len(all_corners)
            judge_agreement = green_num / non_red_num
            self.j1_fraction = j1_fraction*100
            self.j2_fraction = j2_fraction*100
            self.judge_agreement = judge_agreement*100

            all_ra = [i[0][0][0] for i in all_corners]
            all_dec = [i[0][0][1] for i in all_corners]
            mi_ra, ma_ra = min(all_ra), max(all_ra)
            mi_dec, ma_dec = min(all_dec), max(all_dec)
            center_ra = np.mean(all_ra)
            center_dec = np.mean(all_dec)

            ra_dist = SkyCoord(mi_ra, mi_dec, unit='deg').separation(SkyCoord(ma_ra, mi_dec, unit='deg')).arcsecond
            dec_dist = SkyCoord(mi_ra, mi_dec, unit='deg').separation(SkyCoord(mi_ra, ma_dec, unit='deg')).arcsecond

            if ra_dist > dec_dist:
                size = ra_dist/1.375
            else:
                size = dec_dist/1.375
            size = size*1.5

            image_data, header = zoom_image([center_ra, center_dec], size, irsawise_file)
            wcs = WCS(header)

            def pixel_region(region):
                pr = []
                for p in region:
                    sky_c = SkyCoord(p[0],p[1],unit='deg')
                    pixel_p = astropy.wcs.utils.skycoord_to_pixel(sky_c, wcs)
                    pr.append([pixel_p[0], pixel_p[1]])
                return pr

            figure(figsize=(15, 15), dpi=80)
            ax = plt.subplot(projection=wcs)
            for i in range(len(all_corners)):
                region = pixel_region(all_corners[i][0])
                color = all_corners[i][1]
                r = plt.Polygon(region, edgecolor=color, facecolor='none', ls='-', lw=1, zorder=20) 
                ax.add_patch(r)
            
            region = pixel_region(inputs['aperture'])
            r = plt.Polygon(region, edgecolor='pink', facecolor='none', ls='-', lw=1, zorder=19) 
            ax.add_patch(r)

            name = ['SL','SH','LL','LH'][inputs['channel']]+str(suborder)

            plt.annotate(str(inputs['cube_AORKEY'])+'_'+name, xy=(0.05, 0.9), xycoords='axes fraction', size=25, color='orange')
            plt.annotate('J1 cut='+str(inputs['J1_cut']), xy=(0.05, 0.85), xycoords='axes fraction', size=25, color='orange')
            plt.annotate('J2 cut='+str(inputs['J2_cut']), xy=(0.05, 0.8), xycoords='axes fraction', size=25, color='orange')

            plt.annotate('J1$\,$X J2$\,$X', xy=(0.05, 0.25), xycoords='axes fraction', size=25, color='red')
            plt.annotate('J1$\,$X J2$\,\checkmark$', xy=(0.05, 0.2), xycoords='axes fraction', size=25, color='yellow')
            plt.annotate('J1$\,\checkmark$ J2$\,$X', xy=(0.05, 0.15), xycoords='axes fraction', size=25, color='blue')
            plt.annotate('J1$\,\checkmark$ J2$\,\checkmark$', xy=(0.05, 0.1), xycoords='axes fraction', size=25, color='green')

            interval = MinMaxInterval()
            vmin, vmax = interval.get_limits(image_data)
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
            
            plt.imshow(image_data, cmap='gray', zorder=0, norm=norm)
            if show:
                plt.show()
            else:
                return plt
            
        def gold_residuals(self, show=True):
            
            fig, axs = plt.subplots(2, 1, figsize=(12, 12))

            plt.subplot(2, 1, 1)
            gl1, gf1 = self.gold_spec1
            gl2, gf2 = self.gold_spec2
            plt.plot(gl1, gf1, color='gold', label='Gold')
            plt.plot(gl2, gf2, color='gold')
            l1, f1 = self.spec1
            l2, f2 = self.spec2
            plt.plot(l1, f1, color='black', label='SIMLA')
            plt.plot(l2, f2, color='black')
            
            # if inputs['io_correct'] == True:
            #     plt.plot(l1, f1_io, color='black', ls='dashed', label='SIMLA (IO corrected)')
            #     plt.plot(l2, f2_io, color='black', ls='dashed')
            
            plt.legend()
            plt.ylabel('MJy/sr')
            plt.xlabel('micron')

            plt.subplot(2, 1, 2)
            plt.plot(gl1, f1-gf1, color='black')
            plt.plot(gl2, f2-gf2, color='black')
            plt.xlabel('micron')
            plt.ylabel('SIMLA - Gold [MJy/sr]')
            
            if show:
                plt.show()
            else:
                return plt
            
        def stats_sheet(self):
            plt.scatter(np.linspace(0, 10), np.linspace(0, 18), alpha=0)
            plt.text(0, 16, 'Processing time (cubes only): '+str(round(self.cube_time,2))+' s ('+str(round(self.cube_time/60,2))+' mins)')
            plt.text(0, 15, 'Object in cube: '+str(self.object_name))
            plt.text(0, 14, 'Object galactic coords (l, b, [deg]): '+str(self.gal_coords))
            plt.text(0, 13, 'LL1 Center Zodi (MJy/sr): '+str(self.LL1_C_zodi))
            plt.text(0, 12, 'Approx MJD of cube: '+str(round(self.AOR_MJD,2)))
            plt.text(0, 11, 'Number of BCDs in cube: '+str(self.bcds_in_AOR))
            plt.text(0, 10, 'Judge1 Fraction (passed / total): '+str(round(self.j1_fraction,2))+'%')
            plt.text(0, 9, 'Judge2 Fraction (passed / total): '+str(round(self.j2_fraction,2))+'%')
            plt.text(0, 8, 'Judge Agreement (green / non-red): '+str(round(self.judge_agreement,2))+'%')
            plt.text(0, 7, 'AORKEY: '+str(inputs['cube_AORKEY']))
            plt.text(0, 6, 'Order 1 RMS: '+str(round(self.cube_rms[0],2)))
            plt.text(0, 5, 'Order 2 RMS: '+str(round(self.cube_rms[1],2)))
            plt.text(0, 6, 'Order 1 MAD: '+str(round(self.cube_mad[0],2)))
            plt.text(0, 5, 'Order 2 MAD: '+str(round(self.cube_mad[1],2)))
            plt.text(0, 4, 'RAMPTIME: '+str(ramp)+' s')
            plt.text(0, 3, 'SAMPTIME: '+str(samp)+' s')
            plt.text(0, 2, 'Shards selected from this AORKEY (Order1): '+str(self.order1_AOR_shards)+' (Order2): '+str(self.order2_AOR_shards))
            plt.text(0, 1, 'Shards selected NOT from this AORKEY (Order1): '+str(self.order1_nonAOR_shards)+' (Order2): '+str(self.order2_nonAOR_shards))
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            return plt
        
        def zodi_histogram_plot(self):
            plt.hist(self.zodis, bins=20)
            plt.axvline(self.zodi_range[0])
            plt.axvline(self.zodi_range[1])
            plt.axvline(cube_zodi, color='red')
            plt.ylabel('# of Elligible Shards')
            plt.xlabel('Zodi Estimate at LL1 Center (MJy/sr)')
            return plt
        
        def generate_report(self, savename):
            from matplotlib.backends.backend_pdf import PdfPages
            p = PdfPages(savename)
            plt.close()
            self.show_background(show=False).savefig(p, format='pdf')
            plt.close()
            self.show_background_unc(show=False).savefig(p, format='pdf')
            plt.close()
            self.show_depth_map(show=False).savefig(p, format='pdf')
            plt.close()
            self.show_weights(show=False).savefig(p, format='pdf')
            plt.close()
            self.show_dark_spectra(show=False).savefig(p, format='pdf')
            plt.close()
            self.show_extracted_spectra(show=False).savefig(p, format='pdf')
            plt.close()
            self.spectrum_of_background(show=False).savefig(p, format='pdf')
            plt.close()
            self.make_stoplight(1, show=False).savefig(p, format='pdf')
            plt.close()
            self.make_stoplight(2, show=False).savefig(p, format='pdf')
            plt.close()
            self.stats_sheet().savefig(p, format='pdf')
            plt.close()
            self.zodi_histogram_plot().savefig(p, format='pdf')
            plt.close()
            if inputs['isgold']:
                self.gold_residuals(show=False).savefig(p, format='pdf')
            p.close()
            return None
            
        def all_plots(self):
            self.show_background()
            self.show_background_unc()
            self.show_depth_map()
            self.show_weights()
            self.show_dark_spectra()
            self.show_extracted_spectra()
            self.make_stoplight(1)
            self.make_stoplight(2)
            
    return Diagnostics()