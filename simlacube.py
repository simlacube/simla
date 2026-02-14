'''Contains the object class for SIMLA cube objects.'''

import numpy as np
from astropy.stats import sigma_clip
from astropy.io import fits
from idlpy import IDL
import pandas as pd
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy
from shapely import Polygon, Point
from scipy.ndimage import zoom
from astropy.table import Table
import ast
import warnings, sys, os
warnings.filterwarnings("ignore")

from simladb import query, simladbX, DB_bcd, DB_shardpos, \
                    DB_judge1, DB_judge2, DB_foreground, scorners
from simla_utils import fmt_scorners
from simla_variables import SimlaVar

sl_n_shards = SimlaVar().sl_n_shards
ll_n_shards = SimlaVar().ll_n_shards
simlapath = SimlaVar().simlapath
irspath = SimlaVar().irspath

class SimlaCube:

    def __init__(self, aorkey, chnlnum):

        '''Initialize the cube object with important details from the database.'''

        self.AORKEY = aorkey
        self.CHNLNUM = chnlnum

        # use simladbX so that unwanted objects and map types are excluded
        q = query(simladbX.select(DB_bcd.DCEID, DB_bcd.FILE_NAME, DB_bcd.RAMPTIME, \
                                  DB_bcd.OBJECT, DB_bcd.MJD_OBS, DB_bcd.PROGID, DB_bcd.OBJTYPE, \
                                  DB_bcd.STEPSPAR, DB_bcd.STEPSPER, DB_bcd.RA_FOV, DB_bcd.DEC_FOV, \
                                  DB_foreground.ZODI_12, DB_foreground.ISM_12)\
                          .where((DB_bcd.AORKEY==aorkey)&(DB_bcd.CHNLNUM==chnlnum)))

        # look for the unique outputs because tables joined with shard tables will be 
        # duplicated n_shard times
        self.bcd_file_names = irspath+np.unique(q['FILE_NAME'].to_numpy())
        self.dceids = np.unique(q['DCEID'].to_numpy())
        self.RAMPTIME = np.unique(q['RAMPTIME'].to_numpy())[0]
        self.PROGID = np.unique(q['PROGID'].to_numpy())[0]
        self.OBJTYPE = np.unique(q['OBJTYPE'].to_numpy())[0]
        self.IRS_object_name = np.unique(q['OBJECT'].to_numpy())[0]
        self.MJD_mean = np.mean(np.unique(q['MJD_OBS'].to_numpy()))
        self.STEPSPAR = np.unique(q['STEPSPAR'].to_numpy())[0]
        self.STEPSPER = np.unique(q['STEPSPER'].to_numpy())[0]
        self.ZODI_12 = np.unique(q['ZODI_12'].to_numpy())[0]
        self.ISM_12 = np.unique(q['ISM_12'].to_numpy())[0]
        self.ref_coords = (np.nanmean(q['RA_FOV'].to_numpy()), np.nanmean(q['DEC_FOV'].to_numpy()))

        header0 = fits.getheader(self.bcd_file_names[0])
        length = [57, None, 168][self.CHNLNUM]
        width = [3.7, None, 10.7][self.CHNLNUM]

        # Assign a classification based on step size sampling
        # class 1 = has some pixel redundancy 
        # class 2 = no redundancy but no separation
        # class 3 = separation between steps
        percalc = header0['SIZEPER']/width
        if percalc < 1: self.CLASSPER = 1
        elif percalc == 1: self.CLASSPER = 2
        elif percalc > 1: self.CLASSPER = 3

        parcalc = header0['SIZEPAR']/length
        if parcalc < 1: self.CLASSPAR = 1
        elif parcalc == 1: self.CLASSPAR = 2
        elif parcalc > 1: self.CLASSPAR = 3

        # A "SLITLIKE" is a map that is set up like a staring mode observation
        if self.STEPSPER == 1: self.SLITLIKE = True 
        else: self.SLITLIKE = False

    def make_background(self, j1_cut=0.1, j2_cut=2, deltat=5, \
                        zodi_cut=5, ism_cut=0.5, sigma_cut=1.5, \
                        min_shard_depth=50):

        '''
        Method to make the background for the cube object.
        The background is used for all suborders in the channel.

        j1_cut, j2_cut: (float) cuts in MJy/sr for judges. Valid shards are -cut <= val <= cut.
        deltat: (float) the max allowed time difference (days) that a shard can be from the mean
                    time of the build AOR.
        zodi_cut: (float) the max difference in zodi compared with the build AOR (MJy/sr) for shards
                  to qualify as Rank1 or Rank2.
        ism_cut: (float) the max allowed model ISM 12um value.
        sigma_cut: (float) sigma value used to cut out pixels from the background stack.
        min_shard_depth: (int or float) minimum shard depth per shard in a background. This can be exceeded if 
                there are additional shards in-AOR.

        '''

        # A.K.A. Anduril A.K.A. SIMBA

        self.j1_cut = j1_cut
        self.j2_cut = j2_cut
        self.deltat = deltat
        self.zodi_cut = zodi_cut
        self.ism_cut = ism_cut
        self.sigma_cut = sigma_cut
        self.min_shard_depth = min_shard_depth

        # Query the database for potential shards to use as backgrounds
        q = query(simladbX.select(DB_bcd.AORKEY, DB_bcd.DCEID, DB_bcd.FILE_NAME, \
                                  DB_shardpos.SUBORDER, DB_shardpos.SHARD, \
                                  DB_judge1.BACKSUB_PHOT, DB_judge2.F_MEDIAN, \
                                  DB_foreground.ZODI_12, \
                                  DB_bcd.MJD_OBS, DB_bcd.PROGID, DB_bcd.RA_FOV, DB_bcd.DEC_FOV) \
                          .where((DB_bcd.CHNLNUM == self.CHNLNUM) & \
                                 (DB_judge1.BACKSUB_PHOT!=0.0) & \
                                 ((DB_bcd.RAMPTIME > self.RAMPTIME-0.01) & (DB_bcd.RAMPTIME < self.RAMPTIME+0.01)) & \
                                 ((DB_bcd.MJD_OBS >= self.MJD_mean-self.deltat) & \
                                  (DB_bcd.MJD_OBS <= self.MJD_mean+self.deltat)) & \
                                 ((DB_foreground.ZODI_12 >= self.ZODI_12-self.zodi_cut) & \
                                  (DB_foreground.ZODI_12 <= self.ZODI_12+self.zodi_cut)) & \
                                 ((DB_judge1.BACKSUB_PHOT >= -self.j1_cut) & (DB_judge1.BACKSUB_PHOT <= self.j1_cut)) & \
                                 ((DB_judge2.F_MEDIAN >= -self.j2_cut) & (DB_judge2.F_MEDIAN <= self.j2_cut)) & \
                                 (DB_foreground.ISM_12 <= self.ism_cut)))
        aorkeys, dceids, fnames, suborders, shardids, mjds, judge1s, judge2s, zodis, progids, fovras, fovdecs = \
            q['AORKEY'].to_numpy(), q['DCEID'].to_numpy(), irspath+q['FILE_NAME'].to_numpy(), q['SUBORDER'].to_numpy(), \
            q['SHARD'].to_numpy(), q['MJD_OBS'].to_numpy(), q['BACKSUB_PHOT'].to_numpy(), q['F_MEDIAN'].to_numpy(), \
            q['ZODI_12'].to_numpy(), q['PROGID'].to_numpy(), q['RA_FOV'].to_numpy(), q['DEC_FOV'].to_numpy()
        
        mod = ['SL', 'SH', 'LL', 'LH'][self.CHNLNUM]
        self.superdark = np.load(simlapath+'superdarks/tailored_superdarks/'+str(self.AORKEY)+'_'+mod+'.npy')
        self.zodiim = np.load(simlapath+'zodi_images/zodi_images/'+str(self.AORKEY)+'_'+mod+'.npy')

        if self.CHNLNUM == 0: nshards = sl_n_shards
        elif self.CHNLNUM == 2: nshards = ll_n_shards

        ### BEGIN SHARD RANKING AND FINAL SHARD SELECTION ###
        # Rank 1:
        # There are >= min_shard_depth shards within the cube AOR
        # Rank 2:
        # There are = min_shard_depth shards within the zodi and time cut, but other AORs are needed 
        # Rank 3:
        # There are >0 but <min_shard_depth shards within the zodi and time cut
        # Rank 4:
        # No valid shards found, background is only superdark + zodi

        # First, add any in-AOR BCDs
        inaor_condition = (aorkeys == self.AORKEY)

        # Also, attempt to include dedicated backgrounds with a different AOR by including
        # *all* shards where all of the following is true:
        #     - Same PROGID
        #     - Same RAMPTIME
        #     - < 1 day time difference
        #     - < 1 degree separation
        separations = np.degrees(np.arccos(
            (np.sin(np.radians(self.ref_coords[1]))*np.sin(np.radians(fovdecs))) + \
            (np.cos(np.radians(self.ref_coords[1]))*np.cos(np.radians(fovdecs)) * \
             np.cos(np.radians(self.ref_coords[0])-np.radians(fovras)))
        ))
        
        ded_off_condition = (
            (aorkeys != self.AORKEY) & \
            (progids == self.PROGID) & \
            (np.abs(self.MJD_mean - mjds) < 1) & \
            (separations < 1)
        )

        remote_condition = ~(inaor_condition | ded_off_condition)

        rank_string = ''
        final_mask = []
        for suborder in [1, 2]:
            for shardid in range(nshards):

                shard_condition = ((suborders==suborder) & (shardids==shardid))
                
                rank1_mask = np.where(shard_condition & (inaor_condition | ded_off_condition))
                n_rank1 = len(rank1_mask[0])
                this_shard_depth = len(rank1_mask[0])
                final_mask.extend(rank1_mask[0].tolist())
                if n_rank1 >= self.min_shard_depth:
                    rank_string += '1'

                # Now out-of-AOR shards get added in time order
                else:

                    # Find these shards and sort them in time order
                    remote_mask = np.where(shard_condition & remote_condition)
                    remote_mjds = mjds[remote_mask]
                    mjd_diffs = np.abs(self.MJD_mean - remote_mjds)
                    sorted_indices = sorted(range(len(mjd_diffs)), key=lambda i: mjd_diffs[i])
                    sorted_remote_mask = np.asarray([remote_mask[0][i] for i in sorted_indices])

                    # Take only up to min_shard_depth of these
                    trunc_remote_mask = sorted_remote_mask[:self.min_shard_depth-n_rank1]
                    final_mask.extend(trunc_remote_mask)
                    this_shard_depth += len(trunc_remote_mask)

                # Record the rank
                if this_shard_depth == self.min_shard_depth: rank_string += '2'
                elif 0 < this_shard_depth < self.min_shard_depth: rank_string += '3'
                elif this_shard_depth == 0: rank_string += '4'

        # Final selection for shards we are going to use
        final_mask = np.asarray(final_mask)
        self.background_rank = rank_string

        # If no shards are selected, this is the background
        init_background = self.superdark + self.zodiim

        if len(final_mask) > 0:

            # Apply the mask
            aorkeys, dceids, fnames, suborders, shardids, mjds, judge1s, judge2s, zodis = \
                aorkeys[final_mask], dceids[final_mask], fnames[final_mask], \
                suborders[final_mask], shardids[final_mask], mjds[final_mask], \
                judge1s[final_mask], judge2s[final_mask], zodis[final_mask]

            # Load in the masks to select individual shards
            shardmask_lib = [
                [np.load(simlapath+'calib/shard_masks/SL1.npy'), 
                 np.load(simlapath+'calib/shard_masks/SL2.npy'),
                 np.load(simlapath+'calib/shard_masks/SL3.npy')],
                None,
                [np.load(simlapath+'calib/shard_masks/LL1.npy'), 
                 np.load(simlapath+'calib/shard_masks/LL2.npy'),
                 np.load(simlapath+'calib/shard_masks/LL3.npy')],
                None
            ]

            # To select the IO region only
            sl_io_mask = np.where(np.nansum(\
                                   np.nansum(\
                                        [shardmask_lib[0][i] for i in range(3)], \
                                   axis=0), \
                               axis=0)==0, 1, np.nan)
            ll_io_mask = np.where(np.nansum(\
                                   np.nansum(\
                                        [shardmask_lib[2][i] for i in range(3)], \
                                   axis=0), \
                               axis=0)==0, 1, np.nan)
            io_mask = [sl_io_mask, None, ll_io_mask][self.CHNLNUM]

            ### --- ###
            # Here we load in all BCDs, Zodi images, and superdarks that will be used for shards.
            # if caching is implimented in the future, this block will become optional

            # Load all of the unique BCDs that contain qualified shards
            # Recall that in the dceids list, each dceid could be repeated up to n_shards*2 times.
            # For speed, we only want to load in each file once.
            loaded_dceids = np.unique(dceids)
            unique_fnames = np.asarray([fnames[np.where(dceids==d)][0] for d in loaded_dceids])

            loaded_bcd_data = np.asarray([fits.open(i, memmap=False)[0].data for i in unique_fnames])
            loaded_bcd_unc = np.asarray([fits.open(i.replace('bcd.','func.'), memmap=False)[0].data for i in unique_fnames])

            # Do the same for AOR-based files like zodi images and superdarks.
            loaded_aorkeys = np.unique(aorkeys)
            loaded_superdarks = np.asarray([np.load(simlapath+'superdarks/tailored_superdarks/'+ \
                                         str(aorkey)+'_'+mod+'.npy') for aorkey in loaded_aorkeys])
            loaded_zodiims = np.asarray([np.load(simlapath+'zodi_images/zodi_images/'+ \
                                      str(aorkey)+'_'+mod+'.npy') for aorkey in loaded_aorkeys])

            # Pre-combine the calibration data
            loaded_combined_bgs = loaded_zodiims + loaded_superdarks

            # There is a different number of unique DCEIDs and unique AORKEYs. 
            # We need to make a map between the two
            mapped_aors = [aorkeys[np.where(dceids==dceid)][0] for dceid in loaded_dceids]
            mapped_loaded_combined_bgs = np.asarray([loaded_combined_bgs[np.where(loaded_aorkeys==aor)][0] \
                                                         for aor in mapped_aors])

            # Pre-subtract the calibration data. subim_cube has as many planes as there are unique DCEIDs.
            subim_cube = loaded_bcd_data - mapped_loaded_combined_bgs
            ### --- ###

            # When collapsing it, the average is weighted by the number of contributing shards
            io_comp_pops = []
            
            # Loop through the subim_cube and select the masks for qualified shards in each BCD.
            # Add them to the stack.
            shardmask_selection_cube = []
            for dceid in loaded_dceids:
                dceid_master_mask = np.zeros((128, 128)) # container for this BCD
                
                for i in np.where(dceids==dceid)[0]:
                    # one "i" for each qualified shard in this BCD
                    shardmask = shardmask_lib[self.CHNLNUM][suborders[i]-1][shardids[i]]
                    dceid_master_mask += shardmask
                    
                    if suborders[i] == 2:
                        # if a SL2 or LL2 shard qualifies, so does the corresponding SL3 or LL3.
                        bonus_mask = shardmask_lib[self.CHNLNUM][3-1][shardids[i]]
                        dceid_master_mask += bonus_mask

                # Keep track of how many shards from this BCD contributed for the IO pixels
                io_comp_pops.append(len(np.where(dceids==dceid)[0]))
                        
                dceid_master_mask = np.where(dceid_master_mask>1, 1, dceid_master_mask)
                # ^ make sure that overlaps are handled properly
                shardmask_selection_cube.append(dceid_master_mask)

            shardmask_selection_cube = np.asarray(shardmask_selection_cube)
            # shardmask_selection_cube has one plane for each BCD. 1 for pixels on qualified shards, 0 otherwise

            # Use the shardmask_selection_cube to extract the actual BCD data where appropriate
            selected_background_cube = np.where(shardmask_selection_cube==1, subim_cube, np.nan)
            selected_unc_cube = np.where(shardmask_selection_cube==1, loaded_bcd_unc, np.nan)

            # Do the pixel-by-pixel clipping. Since axis=0, pixel values are compared against
            # their peers with the same 2D coordinates
            trimmed_shard_cube = sigma_clip(selected_background_cube, maxiters=3, \
                                            sigma=self.sigma_cut, axis=0, masked=True)
            trimmed_shard_unc_cube = np.where(trimmed_shard_cube.mask, selected_unc_cube, np.nan)

            # Mean-combine the stack and add to the background
            shard_background = np.nanmean(trimmed_shard_cube.data, axis=0)
            background_unc = np.sqrt(np.nansum(trimmed_shard_unc_cube**2, axis=0)) / \
                np.nansum(np.where(trimmed_shard_unc_cube==trimmed_shard_unc_cube, 1, 0), axis=0)

            # Make sure that the IO pixels is 0 in the shard part so that it doesn't negate the IO 
            shard_background = np.where(shard_background!=shard_background, 0, shard_background)

            # Select the IO pixels from contributing BCDs
            io_selection_cube = np.asarray([io_mask for i in subim_cube])
            io_cube = np.where(io_selection_cube==1, subim_cube, np.nan)
            # The IO pixels are averaged over from all BCDs, weighted by the number of contributing shards
            io_weights = np.where(io_cube==io_cube, \
                                  np.asarray(io_comp_pops)[:, np.newaxis, np.newaxis], \
                                  np.nan)
            io_background = np.nansum(io_cube*io_weights, axis=0)/np.nansum(io_weights, axis=0)
            io_background = np.where(io_background!=io_background, 0, io_background)
    
            final_background = init_background + shard_background + io_background

            self.shard_background = shard_background
            self.io_background = io_background
            self.background = final_background
            self.background_unc = background_unc
    
            self.background_depth_map = np.sum(np.where(trimmed_shard_cube.data==trimmed_shard_cube.data, 1, 0), axis=0)
            self.used_shard_data = {'AORKEY': aorkeys, 'DCEID': dceids, 'SUBORDER': suborders, 'SHARD': shardids}

            # Save stats
            self.bg_mean_deltatime = round(np.nanmean(np.abs(mjds-self.MJD_mean)), 5)
            self.bg_mean_deltazodi = round(np.nanmean(np.abs(zodis-self.ZODI_12)), 2)
            self.bg_n_sameaor = np.sum(np.where(aorkeys==self.AORKEY, 1, 0))
            self.bg_n_otheraor = np.sum(np.where(aorkeys!=self.AORKEY, 1, 0))
            self.bg_mean_judge_agreement = np.nanmean(judge1s/judge2s)
            self.mean_background_rank = round(np.mean([int(i) for i in self.background_rank]), 2)

        else:

            # If no shards are found, background is zodi + superdark. Uncertainty is 10%
            self.background = init_background
            self.background_unc = init_background * 0.10
    
            self.background_depth_map = np.zeros((128, 128))
            self.used_shard_data = {'AORKEY': np.asarray([]), 'DCEID': np.asarray([]), \
                                    'SUBORDER': np.asarray([]), 'SHARD': np.asarray([])}

            self.bg_mean_deltazodi = 'N/A'
            self.bg_mean_deltatime = 'N/A'
            self.bg_n_sameaor = 'N/A'
            self.bg_n_otheraor = 'N/A'
            self.bg_mean_judge_agreement = np.nan
            self.mean_background_rank = np.mean([int(i) for i in self.background_rank])

    def build_cube(self, suborder, savename, autobp=True, no_data=True, simlaver='-1'):

        '''
        Wrapper for lights-out IDL code for CUBISM.
        Requires an initialized cube with a background already built.

        suborder: (int) 1, 2, or 3 for the suborder to build.
        savename: (str) the file to save the cube to. Requires ".fits" at the end.
        autobp: (bool) use CUBISM autobadpix?
        no_data: (bool) if False, the .cpj file with be saved with BCD data.
        simlaver: (str) the version of the SIMLA run for the FITS header.

        '''

        self.savename = savename
        self.suborder = suborder
        self.simlaver = simlaver
        
        IDL.run('.RESET_SESSION')

        starting_directory = os.getcwd()
        os.chdir(simlapath)

        IDL.run('.run simla_build')

        IDL.files = self.bcd_file_names.tolist()
        IDL.module = self.CHNLNUM
        IDL.outfile = savename
        IDL.suborder = suborder
        IDL.background = self.background
        IDL.bgunc = self.background_unc

        in_autobp = 1 if autobp==True else 0
        in_nodata = 1 if no_data==True else 0

        IDL.run(f"""simla_build, files, module, outfile, \
                    BACKGROUND_FRAME=background, BACKGROUND_UNC=bgunc, \
                    ORDER=suborder, AUTO_BADPIX={in_autobp}, NO_DATA={in_nodata}""")
        
        os.chdir(starting_directory)

        # There is some issue with the IDL code that names every output .cpj as "unc". This fixes that.
        fixname_cpj = savename.replace('.fits', '.cpj')
        os.system('mv '+savename.replace('.fits', '_unc.cpj')+' '+fixname_cpj)
        self.cpjname = fixname_cpj

        # Update the headers
        self.update_cube_header(simlaver=simlaver)

    def run_sl_io_correct(self, iocorr_savename=None):

        '''
        Run the IDL code by C. Starkey to remove the inter-order artifact from SL cubes.

        iocorr_savename: (str or None) the file name to save the IO-corrected cube to.
                        If None, it is saved with an automatically generated name next to the cube (-iocorr.fits).
        
        '''

        if iocorr_savename is None:
            iocorr_savename = self.savename.replace('.fits', '-iocorr.fits')

        IDL.run('.RESET_SESSION')
        IDL.run('cd, "'+simlapath+'sl_io_correct"')
        IDL.save_location = iocorr_savename
        IDL.cpjpath = self.cpjname
        IDL.run('sl_io_correct, cpjpath, save_location, /QUIET ')
        os.system('cd '+simlapath)

        self.update_cube_header(simlaver=self.simlaver, iocorr=True)

    def update_cube_header(self, simlaver='-1', iocorr=False):

        '''
        Once the cube FITS files have been saved, run this to update the headers

        simlaver: (str) the version of the SIMLA run for the FITS header.
        iocorr: (bool) whether this is an IO-corrected version of the cube.
        '''

        def update_header(header, keywords, values, comments):
            istart = 37
            header.insert(istart, ('', '  / SIMLA PIPELINE'))
            header.insert(istart+1, ('', ''))
            for i in range(len(keywords)):
                header.insert(istart+2+i, (keywords[i], values[i], comments[i]))
            header.insert(istart+len(keywords)+2, ('', ''))
            return header

        keywords = [
            'SIMLAVER', 'MEAN_MJD', 'SLITLIKE', 'CLASSPER', 'CLASSPAR', 'N_BCDS', 
            'ZODI12UM', 'ISM12UM', 
            'BG_RANK', 'BG_DZODI', 'BG_DTIME', 'BG_IN', 'BG_OUT',
            'IOCORR', 
        ]
        values = [
            simlaver, self.MJD_mean, self.SLITLIKE, self.CLASSPER, self.CLASSPAR, \
            len(self.dceids), self.ZODI_12, self.ISM_12, self.mean_background_rank, \
            self.bg_mean_deltazodi, self.bg_mean_deltatime, self.bg_n_sameaor, \
            self.bg_n_otheraor, False,
        ]
        comments = [
            'SIMLA pipeline version',
            '[days] Mean Mod. Julian Date across AOR',
            'True if this cube is staring-mode like',
            'Perpendicular pixel redundancy classification',
            'Parallel pixel redundancy classification',
            'Number of constituent BCDs in cube',
            '[MJy/sr] model zodiacal intensity at 12 micron',
            '[MJy/sr] model ISM intensity at 12 micron',
            'Mean background rank (1=best, 4=worst)',
            '[MJy/sr] mean delta zodi across used BG obs',
            '[days] mean delta time across used BG obs',
            'Number of BG shards from the cube AOR',
            'Number of BG shards NOT from the cube AOR',
            'True if this is IO signal-corrected (SL only)'
        ]

        if not iocorr: savename = self.savename
        elif iocorr: savename = self.savename.replace('.fits', '-iocorr.fits')
        
        cube_hdul = fits.open(savename)
        cube_header = update_header(cube_hdul[0].header, keywords, values, comments)
        if iocorr: cube_header['IOCORR'] = True
        cube_hdul.writeto(savename, overwrite=True)

        uncname = savename.replace('.fits', '_unc.fits')
        unc_hdul = fits.open(uncname)
        unc_header = update_header(unc_hdul[0].header, keywords, values, comments)
        if iocorr: unc_header['IOCORR'] = True
        unc_hdul.writeto(uncname, overwrite=True)

    def save_cpj_params(self, delete_cpj=False, move_to=None):

        '''
        Save various parameters stored in the .cpj files.

        delete_cpj: (bool) if True, delete the cube project file (.cpj) after saving parameters.

        move_to: (None or str) if not None, the path to move output files to (include the final /).

        '''

        cubismpath = SimlaVar().cubismpath

        cpjpath = self.cpjname

        bplpath = self.savename.replace('_cube.fits', '.bpl')
        if move_to is not None: bplpath = move_to+bplpath.split('/')[-1]

        IDL.run('.RESET_SESSION')
        IDL.cpjpath = cpjpath
        IDL.run('.run '+cubismpath+'cubism/cube/cubeproj_load.pro')
        IDL.run('cube=cubeproj_load(cpjpath)')
        IDL.run('cube->SaveBadPixels, "'+bplpath+'"')

        if move_to is not None:
            new_cpjpath = move_to+cpjpath.split('/')[-1]
            os.system('mv '+cpjpath+' '+new_cpjpath)

        if delete_cpj:
            if move_to is None: os.remove(cpjpath)
            elif move_to is not None: os.remove(new_cpjpath)

    def save_background(self, bg_savename=None):

        '''
        Save the built background as a numpy array file.

        bg_savename: (str or None) specify the save name for the background. 
                     If None, it is saved with an automatically generated name next to the cube (_bg.npy).

        '''

        if bg_savename is None: bg_savename = self.savename.replace('_cube.fits', '_bg')
        np.save(bg_savename, self.background)
        np.save(bg_savename.replace('_bg', '_bg_unc'), self.background)

    def save_background_depth_map(self, dmap_name=None):

        '''
        Save the built background depth as a numpy array file.

        dmap_name: (str or None) specify the save name for the background depth map. 
                     If None, it is saved with an automatically generated name next to the cube (_bgdepth.npy).

        '''

        if dmap_name is None: dmap_name = self.savename.replace('_cube.fits', '_bgdepth')
        np.save(dmap_name, self.background_depth_map)

    def save_shardlist(self, shardlist_name=None):

        '''
        Save the list of used shards as a csv file.

        shardlist_name: (str or None) specify the save name for the shard list csv. 
                     If None, it is saved with an automatically generated name next to the cube (_shardlist.csv).

        '''

        if shardlist_name is None: shardlist_name = self.savename.replace('_cube.fits', '_shardlist.csv')
        s_aors, s_dces, s_ids_1, s_ids_2 = [], [], [], []
        for d in sorted(np.unique(self.used_shard_data['DCEID'])):
            s_aors.append(self.used_shard_data['AORKEY'][np.where(self.used_shard_data['DCEID']==d)][0])
            s_dces.append(d)
            s_ids_1.append(str(sorted(self.used_shard_data['SHARD'][np.where((self.used_shard_data['DCEID']==d) & \
                                                                             (self.used_shard_data['SUBORDER']==1))])))
            s_ids_2.append(str(sorted(self.used_shard_data['SHARD'][np.where((self.used_shard_data['DCEID']==d) & \
                                                                             (self.used_shard_data['SUBORDER']==2))])))
        pd.DataFrame({'AORKEY': s_aors,
                      'DCEID': s_dces,
                      'ORDER1_SHARDS': s_ids_1,
                      'ORDER2_SHARDS': s_ids_2}).to_csv(shardlist_name)

    def save_stats(self, statfile_name=None):

        '''
        Save a list of general statistics for the cube and background as a csv file.

        statfile_name: (str or None) specify the save name for the stats csv. 
                     If None, it is saved with an automatically generated name next to the cube (_stats.csv).        

        '''
        
        if statfile_name is None: statfile_name = self.savename.replace('_cube.fits', '_stats.csv')
        pd.DataFrame([{
            'AORKEY': self.AORKEY,
            'CHNLNUM': self.CHNLNUM,
            'SUBORDER': self.suborder,
            'RAMPTIME': self.RAMPTIME,
            'PROGID': self.PROGID,
            'MEAN_MJD': self.MJD_mean,
            'ZODI_12um': self.ZODI_12,
            'ISM_12um': self.ISM_12,
            'N_BCDS': len(self.dceids),
            'CLASSPER': self.CLASSPER, 
            'CLASSPAR': self.CLASSPAR,
            'MEAN_RA': self.ref_coords[0],
            'MEAN_DEC': self.ref_coords[1],
            'OBJNAME': self.IRS_object_name,
            'BG_MEAN_DELTAZODI': self.bg_mean_deltazodi,
            'BG_MEAN_DELTATIME': self.bg_mean_deltatime,
            'BG_RANK': self.background_rank,
            'BG_MEAN_RANK': self.mean_background_rank,
            'BG_N_SAMEAOR': self.bg_n_sameaor,
            'BG_N_OTHERAOR': self.bg_n_otheraor,
        }]).to_csv(statfile_name)

    def make_dark_mask(self, savefile=None, simlaver='-1'):
    
        '''
        Creates a pixel mask for a SIMLA cube corresponding to same-AOR 
        shards used in the background. Dark pixels in this mask have been 
        touched *only* by dark shards.

        NOTE: since we determine whether a pixel is dark based upon whether
        it is in a dark shard, the spatial-direction edges of the darkmask
        are trunated by a pixel relative to the cube footprint. This is probably
        because of the edge-trimming that we did for the shards.

        savefile: (str or None) specify the save name for the map FITS file. 
                     If None, it is saved with an automatically generated name next to the cube (_darkmask.fits).

        simlaver: (str) the version of the SIMLA run for the FITS header.
        '''

        cube_file = self.savename
        suborder = self.suborder

        if suborder == 3: suborder = 2 # bonus order shares the 2nd sub-slit aperture

        if savefile is None: savefile = cube_file.replace('_cube.fits', '_darkmask.fits')

        stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
        # Load in the cube and shardlist
        loadcube = fits.open(cube_file)
        cube_data = loadcube[0].data
        cube_wcs = WCS(loadcube[0].header, fobj=loadcube, naxis=2)
        
        cube_aor, chnlnum = int(loadcube[0].header['AORKEY']), int(loadcube[0].header['CHNLNUM'])
        
        d_aors, d_dceids, d_ids = [], [], []
        for d in sorted(np.unique(self.used_shard_data['DCEID'])):
            d_aors.append(self.used_shard_data['AORKEY'][np.where(self.used_shard_data['DCEID']==d)][0])
            d_dceids.append(d)
            d_ids.append(str(sorted(self.used_shard_data['SHARD'][np.where((self.used_shard_data['DCEID']==d) & \
                                                                           (self.used_shard_data['SUBORDER']==suborder))])))
        d_aors, d_dceids, d_ids = np.asarray(d_aors), np.asarray(d_dceids), np.asarray(d_ids)
        
        # reformat
        d_ids_reform, d_aors_reform, d_dceids_reform = [], [], []
        for i in range(len(d_ids)):
            idlist = np.asarray(ast.literal_eval(d_ids[i]))
            if len(idlist) > 0:
                d_ids_reform.extend(idlist)
                d_aors_reform.extend(d_aors[i]*np.ones_like(idlist))
                d_dceids_reform.extend(d_dceids[i]*np.ones_like(idlist))
        d_ids, d_aors, d_dceids = \
            np.asarray(d_ids_reform), np.asarray(d_aors_reform), np.asarray(d_dceids_reform)
    
        d_ids = d_ids.astype(int)
        d_aors = d_aors.astype(int)
        d_dceids = d_dceids.astype(int)
    
        # Get the dark shards from this AOR
        m_aor = np.where(d_aors==cube_aor)
        d_dceids, d_ids = d_dceids[m_aor], d_ids[m_aor]
        d_pairs = np.asarray([d_dceids, d_ids]).T.tolist()

        if len(d_dceids) > 0:
            
            # Query the DB for the shard corners of dark shards
            q = query(simladbX.select(*scorners, DB_shardpos.DCEID, DB_shardpos.SHARD) \
                      .where((DB_bcd.AORKEY==cube_aor)&(DB_shardpos.CHNLNUM==chnlnum)& \
                             (DB_shardpos.SUBORDER==suborder)))
            corners, q_dceids, q_ids = \
                fmt_scorners(q), q['DCEID'].to_numpy(), q['SHARD'].to_numpy()
            q_pairs = np.asarray([q_dceids, q_ids]).T.tolist()

            # 1 if the shard was used in the BG, otherwise 0
            use_mask = np.asarray([1 if i in d_pairs else 0 for i in q_pairs])
        
            def clip_pixel(x, y):
        
                # return the fraction of the pixel at x,y that is in 
                # the sky aperture
                
                s = 0.5
                x0, y0 = x - s, y - s
                x1, y1 = x - s, y + s
                x2, y2 = x + s, y + s
                x3, y3 = x + s, y - s
                
                pixel_polygon = Polygon([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
                pixel_overlap = region_polygon.intersection(pixel_polygon).area
                normalized_overlap = pixel_overlap / pixel_polygon.area
                
                return normalized_overlap
        
            # Loop through each shard region and see whether each pixel was touched 
            # by that shard
            overlap_cube = []
            for shard in corners:

                try:
        
                    image_xsize = cube_data[0].shape[1]
                    image_ysize = cube_data[0].shape[0]
                    overlap_map = np.zeros_like(cube_data[0])
        
                    # Pixel region of the shard
                    pixel_region = []
                    for p in shard:
                        sky_c = SkyCoord(p[0], p[1], unit='deg')
                        pixel_p = astropy.wcs.utils.skycoord_to_pixel(sky_c, cube_wcs)
                        pixel_region.append([pixel_p[0], pixel_p[1]])
                    region_polygon = Polygon(pixel_region)
                    
                    # Narrow down the clipping area
                    xs = [i[0] for i in pixel_region]
                    ys = [i[1] for i in pixel_region]
                    maxx = int(np.ceil(np.max(xs) + 1))
                    minx = int(np.floor(np.min(xs) - 1))
                    maxy = int(np.ceil(np.max(ys) + 1))
                    miny = int(np.floor(np.min(ys) - 1))
                    coords_to_check = [[x, y]
                                      for x in np.arange(minx, maxx) if 0 <= x <= image_xsize
                                      for y in np.arange(miny, maxy) if 0 <= y <= image_ysize]
        
                    for i in coords_to_check:
                        try:
                            x, y = i[0], i[1]
                            p = Point(x, y)
                            if region_polygon.exterior.distance(p) >= np.sqrt(2)/2 and region_polygon.contains(p):
                                overlap_map[y, x] = 1.0
                            elif region_polygon.exterior.distance(p) <= np.sqrt(2)/2:
                                normalized_overlap = clip_pixel(x, y)
                                overlap_map[y, x] = normalized_overlap
                        except IndexError: pass # Handles regions larger than the cube
            
                    overlap_cube.append(overlap_map)

                except: overlap_cube.append(np.ones_like(cube_data[0])*np.nan)

            # assign a pixel as dark if it was *only* touched by dark shards
            main_overlap_cube = []
            for i in range(len(overlap_cube)):
                main_overlap_cube.append(np.where(overlap_cube[i]>0, use_mask[i], np.nan)) # placeholder must be nan
            main_overlap_cube = np.asarray(main_overlap_cube)
            main_overlap_map = np.nanmin(main_overlap_cube, axis=0)
            # with min, a pixel can only be 1 if it is 1 for all shards

        else: main_overlap_map = np.zeros_like(cube_data[0])

        # Save to FITS
        overlap_header = cube_wcs.to_header()

        cubeheader = loadcube[0].header
        overlap_header.insert(26, ('SIMLAVER', simlaver, 'SIMLA pipeline version'))
        overlap_header.insert(27, ('AORKEY', int(cubeheader['AORKEY']), 'IRS area obeservation request key'))
        overlap_header.insert(28, ('CHNLNUM', cubeheader['CHNLNUM'], 'IRS channel: 0=SL, 1=SH, 2=LL, 3=LH'))
        overlap_header.insert(29, ('APERNAME', cubeheader['APERNAME'], 'IRS module and order'))
        overlap_header.insert(30, ('PROGID', cubeheader['PROGID'], 'IRS Program ID'))
        
        overlap_hdu = fits.PrimaryHDU(data=main_overlap_map, header=overlap_header)
        overlap_hdu.writeto(savefile, overwrite=True)
        
        sys.stdout = stdout

    def save_moment_zero_map(self, mapfile=None, cubefile=None, simlaver='-1'):

        '''
        Save a moment zero (or white light) map of the cube.

        mapfile: (str or None) specify the save name for the map FITS file. 
                     If None, it is saved with an automatically generated name next to the cube (_mom0.fits).   

        cubefile: (str or None) if None, presume the cube associated with SimlaCube.savename
                    if str, give the name of the cube to make a moment zero map for.

        simlaver: (str) the version of the SIMLA run for the FITS header.

        '''

        if cubefile is None: cubefile = self.savename

        if mapfile is None: mapfile = cubefile.replace('_cube.fits', '_mom0.fits')

        cube_data = fits.getdata(cubefile)
        mom0_data = np.nanmean(cube_data, axis=0)
        mom0_data = np.where(mom0_data==0, np.nan, mom0_data)
        
        stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        cubeheader = fits.getheader(cubefile)
        wcs = WCS(cubeheader, fobj=fits.open(cubefile), naxis=2)
        header = wcs.to_header()
        
        mom0_header = wcs.to_header()
        
        mom0_header.insert(26, ('SIMLAVER', simlaver, 'SIMLA pipeline version'))
        mom0_header.insert(27, ('BUNIT', 'MJy/sr', 'Units of surface brightness data'))
        mom0_header.insert(28, ('AORKEY', int(cubeheader['AORKEY']), 'IRS area obeservation request key'))
        mom0_header.insert(29, ('CHNLNUM', cubeheader['CHNLNUM'], 'IRS channel: 0=SL, 1=SH, 2=LL, 3=LH'))
        mom0_header.insert(30, ('APERNAME', cubeheader['APERNAME'], 'IRS module and order'))
        mom0_header.insert(31, ('PROGID', cubeheader['PROGID'], 'IRS Program ID'))
        
        mom0_hdu = fits.PrimaryHDU(data=mom0_data, header=mom0_header)
        mom0_hdu.writeto(mapfile, overwrite=True)
        
        sys.stdout = stdout

    def save_spectrum(self, specfile=None, mask=None, cubefile=None):

        '''
        Save a spectrum from the cube as an IPAC table. The spectrum will be an *average* surface brightness.

        specfile: (str or None) specify the save name for the spectrum .tbl file. 
                     If None, it is saved with an automatically generated name next to the cube (_spec.tbl). 

        mask: (arr or None) if not None, give a numpy array corresponding to pixels to extract
                    for the spectrum. 1=extract, 0=don't extract

        cubefile: (str or None) if None, presume the cube associated with SimlaCube.savename
                    if str, give the name of the cube to make a moment zero map for.

        '''

        if cubefile is None: cubefile = self.savename

        stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        cube = fits.open(cubefile)
        cubedata = cube[0].data
        spectral_axis = cube[1].data[0][0].flatten()

        unc_cube = fits.getdata(cubefile.replace('.fits', '_unc.fits'))

        sys.stdout = stdout

        if mask is None: mask = np.ones_like(cubedata[0])
        mask = np.where(mask==0, np.nan, mask)

        maskcube = np.asarray([mask for i in range(len(cubedata))])
        spectrum = np.nansum(cubedata*maskcube, axis=(1,2))/np.nansum(maskcube, axis=(1,2))
        unc_spectrum = np.sqrt(np.nansum((unc_cube*maskcube)**2, axis=(1,2)))/np.nansum(maskcube, axis=(1,2))

        specdata = pd.DataFrame({
            'lam': spectral_axis,
            'Inu': spectrum,
            'Inu_unc': unc_spectrum,
        })
        
        table_data = Table.from_pandas(specdata)
        
        table_data['lam'].unit = u.micron
        table_data['Inu'].unit = u.MJy/u.sr
        table_data['Inu_unc'].unit = u.MJy/u.sr
        
        if specfile is None: specfile = cubefile.replace('_cube.fits', '_spec.tbl')
        table_data.write(specfile, format='ipac', overwrite=True)

