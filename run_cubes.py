'''
Code to perform a full run of SIMLA cubes.

Configure inputs in simlapath/run_inputs.py. Give the name of the run in run_inputs.py, but the container
directory is set as the runpath in simla_variables.py. The new run will be runpath/run_name. 
The directory for a new run will be made automatically, and the inputs will be copied in there. 

This code takes advantage of the multiprocessing library. Set the number of cores to use in
simla_variables.py.

Prerequisite code: all in database_build/, superdarks/, and zodi_images/, and 
trim_and_shard_masks.py

'''

import numpy as np
import datetime
import time
import os
import gc
import string
import copy
import glob
import pandas as pd
from astropy.io import fits
from multiprocessing import Pool, Process, Queue
from tqdm import tqdm

from simladb import query, DB_bcd
from simla_variables import SimlaVar
from simla_utils import run_inputs_loader, generate_QA_form
from simlacube import SimlaCube

run_start = time.time()

simlapath = SimlaVar().simlapath
runpath = SimlaVar().runpath

# Info for programs that we will make cubes from
q = query(DB_bcd.select(DB_bcd.AORKEY, DB_bcd.DCEID, DB_bcd.FILE_NAME, \
                        DB_bcd.PROGID, DB_bcd.CHNLNUM) \
                .where(((DB_bcd.CHNLNUM==0) | (DB_bcd.CHNLNUM==2)) & \
                       (DB_bcd.AOT_TYPE=='IrsMap') & \
                       (DB_bcd.OBJTYPE.notin(SimlaVar().banned_objtypes)) & \
                       (DB_bcd.OBJECT.notin(SimlaVar().banned_objects)) & \
                       (DB_bcd.AORKEY.notin(SimlaVar().banned_aorkeys))))
aorkeys, dceids, fnames, progids, chnlnums = \
    q['AORKEY'].to_numpy(), q['DCEID'].to_numpy(), q['FILE_NAME'].to_numpy(), \
    q['PROGID'].to_numpy(), q['CHNLNUM'].to_numpy()

# Interprets the txt file containing the inputs
inputs = run_inputs_loader(simlapath+'run_inputs.txt')

# Set up the run directories
run_name = inputs['run_name']
if not os.path.exists(runpath+run_name):
    os.mkdir(runpath+run_name)

productpath = runpath+run_name+'/products/'
ancillarypath = runpath+run_name+'/ancillary/'
if not os.path.exists(productpath):
    os.mkdir(productpath)
if not os.path.exists(ancillarypath):
    os.mkdir(ancillarypath)

simlaver = inputs['simla_version']

# Copy the run inputs into the ancillary directory
os.system('cp '+simlapath+'run_inputs.txt '+ancillarypath+'used_run_inputs.txt')

# Check if there is supposed to be a quality assurance sample
# If so, prepare a list of unique tags for the sample
if SimlaVar().sample_file is not None:
    import pandas as pd
    sample = pd.read_csv(SimlaVar().sample_file)
    sample = [str(sample['AORKEY'][i])+'_'+sample['SUBORDER'][i] for i in range(len(sample['AORKEY']))]
else:
    sample = []

def run_cubes_in_progid(progid):

    # Function so that workers can work on different programs independently.
    
    global build_times

    log_queue.put(str(datetime.datetime.now())+': '+'starting PROGID:'+str(progid))

    prod_progpath = productpath+'PROGID-'+str(progid)+'/'
    anc_progpath = ancillarypath+'PROGID-'+str(progid)+'/'
    if not os.path.exists(prod_progpath):
        os.mkdir(prod_progpath)
    if not os.path.exists(anc_progpath):
        os.mkdir(anc_progpath)

    for aorkey in sorted(np.unique(aorkeys[np.where(progids==progid)])):

        # Try to fix some memory issues.
        gc.collect()

        prod_aorpath = prod_progpath+'AORKEY-'+str(aorkey)+'/'
        anc_aorpath = anc_progpath+'AORKEY-'+str(aorkey)+'/'
        if not os.path.exists(prod_aorpath):
            os.mkdir(prod_aorpath)
        if not os.path.exists(anc_aorpath):
            os.mkdir(anc_aorpath)

        for chnlnum in sorted(np.unique(chnlnums[np.where((progids==progid) & \
                                                          (aorkeys==aorkey))])):

            try:
                mod = ['SL', 'SH', 'LL', 'LH'][chnlnum]

                # Init the cube object
                cube = SimlaCube(aorkey=aorkey, chnlnum=chnlnum)

                # Make the BG, valid for all suborders in this channel
                log_queue.put(str(datetime.datetime.now())+': '+\
                              'making background for '+str(aorkey)+\
                              ', '+mod+'...')
                cube.make_background(
                    j1_cut=inputs['j1_cut'], \
                    j2_cut=inputs['j2_cut'], \
                    deltat=inputs['deltat'], \
                    zodi_cut=inputs['zodi_cut'], \
                    ism_cut=inputs['ism_cut'], \
                    sigma_cut=inputs['sigma_cut'], \
                    min_shard_depth=inputs['min_shard_depth'], \
                )
            except Exception as e:
                log_queue.put(str(datetime.datetime.now())+': background failed (AORKEY='+str(aorkey)+\
                              ')! Error: '+str(e))
                continue
            
            # If the map has multiple targets, need to make multiple cubes
            if cube.OBJTYPE == 'TargetMulti' or cube.OBJTYPE == 'TargetFixedCluster':
        
                map_bcds = []
                map_starting_bcd = 0
                bcdlist = sorted(cube.bcd_file_names)
                while map_starting_bcd < len(bcdlist):
                
                    header = fits.getheader(bcdlist[map_starting_bcd])
                    expected_this_map = header['STEPSPAR'] * header['STEPSPER'] * header['NCYCLES']
                    
                    bcds_this_map = bcdlist[map_starting_bcd: map_starting_bcd+expected_this_map]
                    map_bcds.append(bcds_this_map)
                    
                    map_starting_bcd = map_starting_bcd+expected_this_map

                cubelist, savenames = [], []
                for mapnum in range(len(map_bcds)):
                    
                    subcube = copy.deepcopy(cube)

                    subcube.multi_key = mapnum+1
                    subcube.bcd_file_names = np.asarray(map_bcds[mapnum])
                    savename = str(aorkey)+'-'+str(subcube.multi_key)+'_'+mod
                    
                    cubelist.append(subcube)
                    savenames.append(savename)
                
            else: 
                cubelist = [cube]
                savenames = [str(aorkey)+'-'+str(subcube.multi_key)+'_'+mod]

            for cubeindex in range(len(cubelist)):
                cube = cubelist[cubeindex]

                for suborder in [1, 2, 3]:

                    savename = savenames[cubeindex]+str(suborder)+'_cube.fits'

                    # Set up special treatment for quality assurance cubes
                    in_sample = True if str(aorkey)+'_'+mod+str(suborder) in sample else False
                    no_data = False if in_sample else True
                    delete_cpj = False if in_sample else True
                    
                    try:
                        start = time.time()
                        log_queue.put(str(datetime.datetime.now())+': building cube '+savename+'...')

                        # Now we actually make the cubes
                        cube.build_cube(suborder=suborder, savename=prod_aorpath+savename, \
                                        no_data=no_data, simlaver=simlaver)
                        end = time.time()
                        build_time = round((end-start), 1)
                        log_queue.put(str(datetime.datetime.now())+': successfully built '+savename+' in '+\
                                     str(build_time)+' sec')
                    except Exception as e:
                        log_queue.put(str(datetime.datetime.now())+': cube build failed (AORKEY='+str(aorkey)+\
                                      ')! Error: '+str(e))
                        continue

                    # If SL, run sl_io_correct and save an alternate cube
                    # note: sl_io_correct fails if the number of BCDs in a cube is 1.
                    do_io_correct = False
                    if chnlnum == 0 and len(cube.bcd_file_names) > 1: do_io_correct = True
                        
                    if do_io_correct:
                        try:
                            cube.run_sl_io_correct()
                            log_queue.put(str(datetime.datetime.now())+': successfully ran IO correct for '+\
                                          savename+' in '+\
                                          str(build_time)+' sec')
                        except Exception as e:
                            log_queue.put(str(datetime.datetime.now())+': error running IO correct for AORKEY='+\
                                          str(aorkey)+'. Error: '+str(e))
                            continue
                    else:
                        log_queue.put(str(datetime.datetime.now())+': did not run sl_io_correct for AORKEY='+\
                                      str(aorkey)+'. Number of BCDs=1.')

                    try:
                        # Saving additional information
                        cube.save_cpj_params(delete_cpj=delete_cpj, move_to=anc_aorpath) # .cpj files take a lot of storage!
                        cube.save_background(bg_savename=anc_aorpath+savename.replace('_cube.fits', '_bg'))
                        cube.save_background_depth_map(dmap_name=anc_aorpath+savename.replace('_cube.fits', '_bgdepth'))
                        cube.save_shardlist(shardlist_name=anc_aorpath+savename.replace('_cube.fits', '_shardlist.csv'))
                        cube.save_stats(statfile_name=anc_aorpath+savename.replace('_cube.fits', '_stats.csv'))
                    except Exception as e:
                        log_queue.put(str(datetime.datetime.now())+': error saving additional information for AORKEY='+\
                                      str(aorkey)+'. Error: '+str(e))
                        continue

                    try:
                        # Save the non-cube deliverable data products
                        cube.make_dark_mask(simlaver=simlaver)
                        cube.save_moment_zero_map(simlaver=simlaver)
                        cube.save_spectrum()
                        if do_io_correct: 
                            cube.save_spectrum(specfile=prod_aorpath+savename.replace('_cube.fits', '_spec-iocorr.tbl'), 
                                               cubefile=prod_aorpath+savename.replace('.fits', '-iocorr.fits'))
                            
                    except Exception as e:
                        log_queue.put(str(datetime.datetime.now())+': error saving non-cube deliverable products for AORKEY='+\
                                      str(aorkey)+'. Error: '+str(e))
                        continue

                    try:
                        qa_savename_template = anc_aorpath+savename.replace('_cube.fits', '')
                        
                        # Create the quality assurance PDF
                        darkmask = fits.getdata(cube.savename.replace('_cube.fits', '_darkmask.fits'))
                        brightmask = np.where(darkmask==0, 1, 0)
                        
                        cube.save_spectrum(specfile=qa_savename_template+'_darkspec.tbl', \
                                           mask=darkmask)
                        cube.save_spectrum(specfile=qa_savename_template+'_brightspec.tbl', \
                                           mask=brightmask)
                        
                        if do_io_correct:
                            cube.save_spectrum(cubefile=cube.savename.replace('_cube.fits', '_cube-iocorr.fits'), \
                                               specfile=qa_savename_template+'_darkspec-iocorr.tbl', \
                                               mask=darkmask)
                            cube.save_spectrum(cubefile=cube.savename.replace('_cube.fits', '_cube-iocorr.fits'), \
                                               specfile=qa_savename_template+'_brightspec-iocorr.tbl', \
                                               mask=brightmask)
                            qa_iocorr = True
                        else: qa_iocorr = False

                        generate_QA_form(cube.savename, anc_aorpath, qa_savename_template+'_QAplots.pdf', \
                                         iocorr_spectra=qa_iocorr)

                    except Exception as e:
                        log_queue.put(str(datetime.datetime.now())+': error creating QA PDF for AORKEY='+\
                                      str(aorkey)+'. Error: '+str(e))
                        continue

# These are necessary for the "workers" to be able to write the log without collisions
def write_log(queue):
    log_fname = runpath+run_name+'/log.txt'
    with open(log_fname, "w") as f:
        for msg in iter(queue.get, None):
            f.write(msg + "\n")
            f.flush()
log_queue = None
def init_worker(queue):
    global log_queue
    log_queue = queue

# Initialize the workers and run
ps = np.unique(progids)
cal_progids = np.genfromtxt(SimlaVar().simlapath+'calib/SIRTFcal_progids.txt')
ps = np.asarray([i for i in ps if i not in cal_progids]) # exclude SIRTF Calibration programs
if __name__ == '__main__':

    log_queue = Queue()
    logger = Process(target=write_log, args=(log_queue,))
    logger.start()

    with Pool(processes=SimlaVar().processors, initializer=init_worker, initargs=(log_queue,)) as pool: 
        for _ in tqdm(pool.imap_unordered(run_cubes_in_progid, ps), total=len(ps)):
            pass
    
    log_queue.put(str(datetime.datetime.now())+': compiling cube stats CSV.')
    statsfiles = glob.glob(ancillarypath+'/**/**/*stats.csv')
    master_csv = pd.concat((pd.read_csv(f) for f in statsfiles), ignore_index=True)
    master_csv.to_csv(productpath+'/cube_stats_'+simlaver+'.csv', index=False)

    run_end = time.time()
    log_queue.put('Done! This run took '+str(round((run_end-run_start)/3600, 2))+'hrs to complete.')

    log_queue.put(None)
    logger.join()
