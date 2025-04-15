'''Contains the object class for SIMLA cube objects.'''

import numpy as np
from astropy.stats import sigma_clip
from astropy.io import fits
import os
from idlpy import IDL
import pandas as pd

from simladb import query, simladbX, DB_bcd, DB_shardpos, \
                    DB_judge1, DB_judge2, DB_foreground
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
                                  DB_bcd.OBJECT, DB_bcd.MJD_OBS, \
                                  DB_bcd.STEPSPAR, DB_bcd.STEPSPER, \
                                  DB_foreground.ZODI_12, DB_foreground.ISM_12)\
                          .where((DB_bcd.AORKEY==aorkey)&(DB_bcd.CHNLNUM==chnlnum)))

        # look for the unique outputs because tables joined with shard tables will be 
        # duplicated n_shard times
        self.bcd_file_names = irspath+np.unique(q['FILE_NAME'].to_numpy())
        self.dceids = np.unique(q['DCEID'].to_numpy())
        self.RAMPTIME = np.unique(q['RAMPTIME'].to_numpy())[0]
        self.IRS_object_name = np.unique(q['OBJECT'].to_numpy())[0]
        self.MJD_mean = np.mean(np.unique(q['MJD_OBS'].to_numpy()))
        self.STEPSPAR = np.unique(q['STEPSPAR'].to_numpy())[0]
        self.STEPSPER = np.unique(q['STEPSPER'].to_numpy())[0]
        self.ZODI_12 = np.unique(q['ZODI_12'].to_numpy())[0]
        self.ISM_12 = np.unique(q['ISM_12'].to_numpy())[0]

    def make_background(self, j1_cut=0.1, j2_cut=2, deltat=5, max_deltat=10, \
                        zodi_cut=5, ism_cut=0.5, sigma_cut=1.5, \
                        desired_shard_depth=100, use_io_correct=False):

        '''
        Method to make the background for the cube object.
        The background is used for all suborders in the channel.

        j1_cut, j2_cut: (float) cuts in MJy/sr for judges. Valid shards are -cut <= val <= cut.
        deltat: (float) the max difference in time from the mean time of the build AOR (days) for shards
                to qualify as Rank1.
        max_deltat: (float) the max allowed time difference (days) that a shard can be from the mean
                    time of the build AOR.
        zodi_cut: (float) the max difference in zodi compared with the build AOR (MJy/sr) for shards
                  to qualify as Rank1 or Rank2.
        ism_cut: (float) the max allowed model ISM 12um value.
        sigma_cut: (float) sigma value used to cut out pixels from the background stack.
        desired_shard_depth: (int or float) target shard depth used for determining background ranks.
        use_io_correct: (bool) whether to apply io_correct (not supported yet).

        '''

        # A.K.A. Anduril A.K.A. SIMBA

        self.j1_cut = j1_cut
        self.j2_cut = j2_cut
        self.deltat = deltat
        self.max_deltat = max_deltat
        self.zodi_cut = zodi_cut
        self.ism_cut = ism_cut
        self.sigma_cut = sigma_cut
        self.desired_shard_depth = desired_shard_depth
        self.use_io_correct = use_io_correct

        # Query the database for potential shards to use as backgrounds
        q = query(simladbX.select(DB_bcd.AORKEY, DB_bcd.DCEID, DB_bcd.FILE_NAME, \
                                  DB_shardpos.SUBORDER, DB_shardpos.SHARD, \
                                  DB_bcd.MJD_OBS, DB_foreground.ZODI_12) \
                          .where((DB_bcd.CHNLNUM == self.CHNLNUM) & \
                                 (DB_judge1.BACKSUB_PHOT!=0.0)&\
                                 ((DB_bcd.RAMPTIME > self.RAMPTIME-0.01) & (DB_bcd.RAMPTIME < self.RAMPTIME+0.01)) & \
                                 ((DB_bcd.MJD_OBS >= self.MJD_mean-self.max_deltat) & (DB_bcd.MJD_OBS <= self.MJD_mean+self.max_deltat)) & \
                                 ((DB_judge1.BACKSUB_PHOT >= -self.j1_cut) & (DB_judge1.BACKSUB_PHOT <= self.j1_cut)) & \
                                 ((DB_judge2.F_MEDIAN >= -self.j2_cut) & (DB_judge2.F_MEDIAN <= self.j2_cut)) & \
                                 (DB_foreground.ISM_12 <= self.ism_cut)))
        aorkeys, dceids, fnames, suborders, shardids, mjds, zodis = \
            q['AORKEY'].to_numpy(), q['DCEID'].to_numpy(), irspath+q['FILE_NAME'].to_numpy(), q['SUBORDER'].to_numpy(), \
            q['SHARD'].to_numpy(), q['MJD_OBS'].to_numpy(), q['ZODI_12'].to_numpy()

        mod = ['SL', 'SH', 'LL', 'LH'][self.CHNLNUM]
        self.superdark = np.load(simlapath+'superdarks/tailored_superdarks/'+str(self.AORKEY)+'_'+mod+'.npy')
        self.zodiim = np.load(simlapath+'zodi_images/zodi_images/'+str(self.AORKEY)+'_'+mod+'.npy')

        if self.CHNLNUM == 0: nshards = sl_n_shards
        elif self.CHNLNUM == 2: nshards = ll_n_shards

        ### BEGIN SHARD RANKING AND FINAL SHARD SELECTION ###
        # Rank 1:
        # There are >= desired_shard_depth shards within delta_t and the zodi cut
        r1_condition = ((np.abs(mjds - self.MJD_mean) <= self.deltat) & \
                         (np.abs(zodis - self.ZODI_12) <= self.zodi_cut))

        # Rank 2:
        # There >= desired_shard_depth shards within the zodi cut, 
        # but we take the nearest in time shards outside of deltat 
        # but difference in time is < max_deltat
        r2_condition = ((np.abs(zodis - self.ZODI_12) <= self.zodi_cut))

        # Rank 3:
        # After removing zodi cut and deltat cut, there are >= desired_shard_depth shards
        # but difference in time is < max_deltat

        # Rank 4:
        # The background gets some shards but < desired_shard_depth

        # Rank 5:
        # No valid shards found, background is only superdark + zodi

        rank_string = ''
        final_mask = []
        for suborder in [1, 2]:
            for shardid in range(nshards):
                shard_condition = ((suborders==suborder) & (shardids==shardid))

                mask_rank1 = np.where(r1_condition & shard_condition)
                mask_rank2 = np.where(r2_condition & shard_condition)
                
                if len(dceids[mask_rank1]) >= self.desired_shard_depth:
                    final_mask.extend(mask_rank1[0].tolist())
                    rank_string += '1'

                elif len(dceids[mask_rank2]) >= self.desired_shard_depth:
                    final_mask.extend(mask_rank2[0].tolist())
                    rank_string += '2'

                else:

                    mask_rank3thru5 = np.where(shard_condition)
                    final_mask.extend(mask_rank3thru5[0].tolist())
                    this_n = len(dceids[mask_rank3thru5])
                
                    if this_n >= self.desired_shard_depth: rank_string += '3'
                    elif 0 < this_n < self.desired_shard_depth: rank_string += '4'
                    elif this_n == 0: rank_string += '5'

        # Final selection for shards we are going to use
        final_mask = np.asarray(final_mask)
        self.background_rank = rank_string

        # If no shards are selected, this is the background
        init_background = self.superdark + self.zodiim

        if len(final_mask) > 0:

            # Apply the mask
            aorkeys, dceids, fnames, suborders, shardids, mjds = \
                aorkeys[final_mask], dceids[final_mask], fnames[final_mask], \
                suborders[final_mask], shardids[final_mask], mjds[final_mask]

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
    
            final_background = init_background + shard_background
            
            self.background = final_background
            self.background_unc = background_unc
    
            self.background_depth_map = np.sum(np.where(trimmed_shard_cube.data==trimmed_shard_cube.data, 1, 0), axis=0)
            self.used_shard_data = {'AORKEY': aorkeys, 'DCEID': dceids, 'SUBORDER': suborders, 'SHARD': shardids}

        else:

            # If no shards are found, background is zodi + superdark. Uncertainty is 10%
            self.background = init_background
            self.background_unc = init_background * 0.10
    
            self.background_depth_map = np.zeros((128, 128))
            self.used_shard_data = {'AORKEY': np.asarray([]), 'DCEID': np.asarray([]), 'SUBORDER': np.asarray([]), 'SHARD': np.asarray([])}

    def build_cube(self, suborder, savename, autobp=True):

        '''
        Wrapper for lights-out IDL code for CUBISM.
        Requires an initialized cube with a background already built.

        suborder: (int) 1, 2, or 3 for the suborder to build
        savename: (str) the file to save the cube to. Requires ".fits" at the end.
        autobp: (bool) use CUBISM autobadpix?

        '''

        self.savename = savename
        
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

        if autobp:
            IDL.run('simla_build, files, module, outfile, \
                        BACKGROUND_FRAME=background, BACKGROUND_UNC=bgunc, \
                        ORDER=suborder, /AUTO_BADPIX')
        
        else:
            IDL.run('simla_build, files, module, outfile, \
                        BACKGROUND_FRAME=background, BACKGROUND_UNC=bgunc, \
                        ORDER=suborder') 
        
        os.chdir(starting_directory)

        # There is some issue with the IDL code that names every output .cpj as "unc". This fixes that.
        os.system('mv '+savename.replace('.fits', '_unc.cpj')+' '+savename.replace('.fits', '.cpj'))

    def save_cpj_params(self, delete_cpj=False):

        '''
        Save various parameters stored in the .cpj files.

        delete_cpj: (bool) if True, delete the cube project file (.cpj) after saving parameters.

        '''

        cubismpath = SimlaVar().cubismpath

        IDL.run('.RESET_SESSION')
        IDL.cpjpath = self.savename.replace('.fits', '.cpj')
        IDL.run('.run '+cubismpath+'cubism/cube/cubeproj_load.pro')
        IDL.run('cube=cubeproj_load(cpjpath)')
        IDL.run('cube->SaveBadPixels, "'+self.savename.replace('.fits', '.bpl')+'"')

        if delete_cpj: os.remove(self.savename.replace('.fits', '.cpj'))

    def save_background(self, bg_savename=None):

        '''
        Save the built background as a numpy array file.

        bg_savename: (str or None) specify the save name for the background. 
                     If None, it is saved with an automatically generated name next to the cube (_bg.npy).

        '''

        if bg_savename is None: bg_savename = self.savename.replace('.fits', '_bg')
        np.save(bg_savename, self.background)
        np.save(bg_savename.replace('_bg', '_bg_unc'), self.background)

    def save_background_depth_map(self, dmap_name=None):

        '''
        Save the built background depth as a numpy array file.

        dmap_name: (str or None) specify the save name for the background depth map. 
                     If None, it is saved with an automatically generated name next to the cube (_bgdepth.npy).

        '''

        if dmap_name is None: dmap_name = self.savename.replace('.fits', '_bgdepth')
        np.save(dmap_name, self.background_depth_map)

    def save_shardlist(self, shardlist_name=None):

        '''
        Save the list of used shards as a csv file.

        shardlist_name: (str or None) specify the save name for the shard list csv. 
                     If None, it is saved with an automatically generated name next to the cube (_shardlist.csv).

        '''

        if shardlist_name is None: shardlist_name = self.savename.replace('.fits', '_shardlist.csv')
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
        

        

        





















        