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
from astropy.io import fits
from multiprocessing import Pool, Process, Queue

from simladb import query, DB_bcd
from simla_variables import SimlaVar
from simla_utils import run_inputs_loader
from simlacube import SimlaCube

run_start = time.time()

simlapath = SimlaVar().simlapath
runpath = SimlaVar().runpath

# Mapping between the PROGID and the name of the run. That is not in the DB.
prognames_map = np.genfromtxt(simlapath+'storage/prognames.txt', dtype=str).T # [names, progids]
prognames = prognames_map[0]
valid_progids = prognames_map[1].astype(int) # we will only build cubes from this file.

# Info for programs that we will make cubes from
q = query(DB_bcd.select(DB_bcd.AORKEY, DB_bcd.DCEID, DB_bcd.FILE_NAME, \
                        DB_bcd.PROGID, DB_bcd.OBJECT, DB_bcd.CHNLNUM) \
                .where(((DB_bcd.CHNLNUM==0) | (DB_bcd.CHNLNUM==2)) & \
                       (DB_bcd.AOT_TYPE=='IrsMap') & \
                       (DB_bcd.OBJTYPE.notin(SimlaVar().banned_objtypes)) & \
                       (DB_bcd.OBJECT.notin(SimlaVar().banned_objects)) & \
                       (DB_bcd.AORKEY.notin(SimlaVar().banned_aorkeys))))
aorkeys, dceids, fnames, progids, objects, chnlnums = \
    q['AORKEY'].to_numpy(), q['DCEID'].to_numpy(), q['FILE_NAME'].to_numpy(), \
    q['PROGID'].to_numpy(), q['OBJECT'].to_numpy(), q['CHNLNUM'].to_numpy()

# Interprets the txt file containing the inputs
inputs = run_inputs_loader(simlapath+'run_inputs.txt')

# Set up the run directory
run_name = inputs['run_name']
if not os.path.exists(runpath+run_name):
    os.mkdir(runpath+run_name)

# Copy the run inputs into the new directory
os.system('cp '+simlapath+'run_inputs.txt '+runpath+run_name+'/used_run_inputs.txt')

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

    # Make sure that we are only building programs that we want to be
    if progid in valid_progids:

        progname = prognames[np.where(valid_progids==progid)][0]

        log_queue.put(str(datetime.datetime.now())+': '+'starting PROGID:'+str(progid)+' ('+progname+')')

        progpath = runpath+run_name+'/'+progname+'_PROGID'+str(progid)+'/'
        if not os.path.exists(progpath):
            os.mkdir(progpath)
    
        for objname in np.unique(objects[np.where(progids==progid)]):

            # Replace any characters that will cause problems
            fixed_objname = objname.replace(' ','_') \
                                   .replace('/','-') \
                                   .replace('(','_') \
                                   .replace(')','_') \
                                   .replace(',','_') \
                                   .replace('&','_')
            objpath = progpath+fixed_objname+'/'
            if not os.path.exists(objpath):
                os.mkdir(objpath)
    
            for aorkey in sorted(np.unique(aorkeys[np.where((progids==progid) & \
                                                            (objects==objname))])):

                # Try to fix some memory issues.
                gc.collect()
        
                aorpath = objpath+str(aorkey)+'/'
                if not os.path.exists(aorpath):
                    os.mkdir(aorpath)
        
                for chnlnum in sorted(np.unique(chnlnums[np.where((progids==progid) & \
                                                                  (aorkeys==aorkey))])):

                    try:
                        mod = ['SL', 'SH', 'LL', 'LH'][chnlnum]

                        # Init the cube object
                        cube = SimlaCube(aorkey=aorkey, chnlnum=chnlnum)

                        # Make the BG, valid for all suborders in this channel
                        log_queue.put(str(datetime.datetime.now())+': '+\
                                      'making background for '+str(aorkey)+\
                                      ', '+fixed_objname+', '+mod+'...')
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
                        sample_header = fits.getheader(cube.bcd_file_names[0])
                        n_fovid = len(np.unique([fits.getheader(i)['FOVID'] for i in cube.bcd_file_names]))
                        nbcd_per_map = sample_header['STEPSPAR'] * sample_header['STEPSPER'] * n_fovid 
                        n_unique_bcds = len(cube.bcd_file_names) / sample_header['NCYCLES']
                        n_maps = int(n_unique_bcds / nbcd_per_map)
                        nbcd_per_map = nbcd_per_map * sample_header['NCYCLES']
                        sorted_bcds = sorted(cube.bcd_file_names)
                        letters = list(string.ascii_uppercase)
                        cubelist, savenames = [], []
                        for mapnum in range(n_maps):
                            bcds_in_map = sorted_bcds[mapnum*nbcd_per_map: (mapnum*nbcd_per_map)+nbcd_per_map]
                            subcube = copy.deepcopy(cube)
                            subcube.bcd_file_names = np.asarray(bcds_in_map)
                            savename = str(aorkey)+letters[mapnum]+'_'+fixed_objname+'_'+mod
                            cubelist.append(subcube)
                            savenames.append(savename)
                    else: 
                        cubelist = [cube]
                        savenames = [str(aorkey)+'_'+fixed_objname+'_'+mod]

                    for cubeindex in range(len(cubelist)):
                        cube = cubelist[cubeindex]
    
                        for suborder in [1, 2, 3]:

                            savename = savenames[cubeindex]+str(suborder)+'.fits'
    
                            # Set up special treatment for quality assurance cubes
                            in_sample = True if str(aorkey)+'_'+mod+str(suborder) in sample else False
                            no_data = False if in_sample else True
                            delete_cpj = False if in_sample else True
                            
                            try:
                                start = time.time()
                                log_queue.put(str(datetime.datetime.now())+': building cube '+savename+'...')
    
                                # Now we actually make the cubes
                                cube.build_cube(suborder=suborder, savename=aorpath+savename, no_data=no_data)
                                end = time.time()
                                build_time = round((end-start), 1)
                                log_queue.put(str(datetime.datetime.now())+': successfully built '+savename+' in '+\
                                             str(build_time)+' sec')
                            except Exception as e:
                                log_queue.put(str(datetime.datetime.now())+': cube build failed (AORKEY='+str(aorkey)+\
                                              ')! Error: '+str(e))
                                continue
    
                            # If SL, run sl_io_correct and save an alternate cube
                            if chnlnum == 0:
                                try:
                                    cube.run_sl_io_correct()
                                    log_queue.put(str(datetime.datetime.now())+': successfully ran IO correct for '+\
                                                  savename+' in '+\
                                                  str(build_time)+' sec')
                                except Exception as e:
                                    log_queue.put(str(datetime.datetime.now())+': error running IO correct for AORKEY='+\
                                                  str(aorkey)+'. Error: '+str(e))
                                    continue
    
                            try:
                                # Saving additional information
                                cube.save_cpj_params(delete_cpj=delete_cpj) # .cpj files take a lot of storage!
                                cube.save_background()
                                cube.save_background_depth_map()
                                cube.save_shardlist()
                                cube.save_stats()
                            except Exception as e:
                                log_queue.put(str(datetime.datetime.now())+': error saving non-cube products for AORKEY='+\
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
if __name__ == '__main__':

    log_queue = Queue()
    logger = Process(target=write_log, args=(log_queue,))
    logger.start()
    
    with Pool(processes=SimlaVar().processors, initializer=init_worker, initargs=(log_queue,)) as pool: 
        for _ in pool.imap_unordered(run_cubes_in_progid, ps):
            pass

    run_end = time.time()
    log_queue.put('Done! This run took '+str(round((run_end-run_start)/3600, 2))+'hrs to complete.')

    log_queue.put(None)
    logger.join()

    