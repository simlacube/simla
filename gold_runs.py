import numpy as np
import os
import glob
from simlacube import simlacube
from simla import tools
import pickle
from simla.database import query, bcd, stats, baby, qFull, connect, zodi,\
    wmap, qAll, shardphot, galcoords, subspec, qShard, pq, shardpos, judge2
import datetime
import ast

# This is the same as run_cubes.py, but for only (a subsample of) the golds

def gold_run(delta_t, J1_cut, J2_cut, zodi_cut, io_correct, superdark, in_aor):

    gold_only = True
    new_rootdir = '/home/work/simla/test_cubes/gold_tests/'

    run_name = 'dt='+str(delta_t)+\
            '_J1='+str(J1_cut)+\
            '_J2='+str(J2_cut)+\
            '_zc='+str(zodi_cut)+\
            '_io='+str(io_correct)+\
            '_sd='+str(superdark)+\
            '_ia='+str(in_aor)+\
            '/'

    if not os.path.exists(new_rootdir+run_name):
        os.mkdir(new_rootdir+run_name)
        print('RUN: '+run_name)
    else: 
        print('SKIPPING: '+run_name+' (already done)')
        return

    simlapath = os.path.dirname(os.path.realpath(__file__))

    prognames_map = p = np.genfromtxt(simlapath+'/prognames.txt', dtype=str)
    prognames = prognames_map.T[0]
    progids = prognames_map.T[1]

    all_aors = np.load(simlapath+'/gold_aors.npy')

    all_aors = np.unique(all_aors).tolist() # Remove [0::2] 
    aor_counter = -1

    if not os.path.exists(new_rootdir+run_name+'diagnostics'):
        os.system('mkdir '+new_rootdir+run_name+'diagnostics')

    log_fname = new_rootdir+run_name+'log.txt'
    failed_cubes_fname = new_rootdir+run_name+'failed_cubes.txt'

    for i in range(len(progids)):

        progid = progids[i]
        progname = prognames[i]

        aors_in_progid = query(bcd.select(bcd.AORKEY).where(bcd.PROGID==int(progid))\
                     .where(bcd.AOT_TYPE=='IrsMap')\
                     .where(bcd.OBJTYPE!='TargetMulti'))['AORKEY'].to_numpy()
        aors_in_progid = np.unique(aors_in_progid).tolist()
        
        aors_in_progid = [aor for aor in aors_in_progid if aor in all_aors]
        
        if len(aors_in_progid) > 0:

            progdir = new_rootdir+run_name+progname+'_PROGID'+progid+'/'
            if not os.path.exists(progdir):
                os.system('mkdir '+progdir)

            for aor in aors_in_progid:

                has_sl = True if len(query(bcd.select(bcd.DCEID).where(bcd.AORKEY==aor)\
                                       .where(bcd.CHNLNUM==0))['DCEID'].to_numpy()) > 0 else False

                has_ll = True if len(query(bcd.select(bcd.DCEID).where(bcd.AORKEY==aor)\
                                       .where(bcd.CHNLNUM==2))['DCEID'].to_numpy()) > 0 else False

                if has_sl or has_ll:

                    log = []
                    failed_cubes = []

                    if aors_in_progid.index(aor) == 0:
                        log.append(str(datetime.datetime.now())+' : '+'beginning program: '+\
                                   progname+' (PROGID: '+progid+')')

                    aor_counter += 1

                    o = query(bcd.select(bcd.OBJECT).where(bcd.AORKEY==aor))['OBJECT'][0]
                    o = o.replace(' ','_').replace('/','-').replace('(','_').replace(')','_').replace(',','_')
                    if not os.path.exists(progdir+o):
                        os.system('mkdir '+progdir+o)

                    aordir = progdir+o+'/'+str(aor)
                    if not os.path.exists(aordir):
                        os.system('mkdir '+aordir)

                    log.append(str(datetime.datetime.now())+' : '+'now processing AORKEY: '+str(aor)+', OBJECT: '+o)
                    log.append(str(datetime.datetime.now())+' : '+'this is aor '+str(aor_counter)+'/'+str(len(all_aors)))

                    inputs = {
                        'cube_AORKEY': aor,
                        'channel': 0,
                        'delta_t': delta_t,
                        'J1_cut': J1_cut, 
                        'J2_cut': J2_cut,
                        'savename': aordir+'/'+str(aor)+'_'+o,
                        'dev': True,
                        'aperture': 'standard',
                        'isgold': False,
                        'gold_cube': None,
                        'io_correct': io_correct,
                        'zodi_cut': zodi_cut,
                        'superdark': superdark,
                        'ignore_in_AORKEY': in_aor
                    }

                    if not has_sl: log.append(str(datetime.datetime.now())+' : '+'no SL in this AORKEY')

                    if has_sl:
                        log.append(str(datetime.datetime.now())+' : '+'making SL cubes...')
                        try:

                            sl_info = simlacube(inputs)
                            log.append(str(datetime.datetime.now())+' : '+'successfully made SL cubes in '+str(round(sl_info.cube_time,2))+'s')

                        except Exception as e: 
                            log.append(str(datetime.datetime.now())+' : '+'ERROR: '+str(e))
                            failed_cubes.append(str(aor)+' SL')
                            pass

                        try:

                            log.append(str(datetime.datetime.now())+' : '+'now preparing the diagnostics...')
                            diag_name = new_rootdir+run_name+'diagnostics/'+o+'_'+str(aor)+'_'+'_deltat='+str(delta_t)+'_J1='+\
                                                    str(J1_cut)+'_J2='+str(J2_cut)+'_zodicut='+str(zodi_cut)+'_SL_diag'
                            sl_info.generate_report(diag_name+'.pdf')
                            sl_diag_data = {
                            'AORKEY': aor,
                            'cube_time': round(sl_info.cube_time,2),
                            'object': sl_info.object_name,
                            'galactic_coords': sl_info.gal_coords,
                            'zodi_estimate': sl_info.LL1_C_zodi,
                            'mjd': sl_info.AOR_MJD,
                            'number_of_BCDs': sl_info.bcds_in_AOR,
                            'J1_fraction': sl_info.j1_fraction,
                            'J2_fraction': sl_info.j2_fraction,
                            'judge_agreement': sl_info.judge_agreement,
                            'SL1_RMS': sl_info.cube_rms[0],
                            'SL2_RMS': sl_info.cube_rms[1],
                            'SL1_MAD': sl_info.cube_mad[0],
                            'SL2_MAD': sl_info.cube_mad[1],
                            'RAMPTIME': sl_info.RAMPTIME,
                            'SAMPTIME': sl_info.SAMPTIME,
                            'EXPTOT_T': sl_info.EXPTOT_T,
                            'CHNLNUM': 0,
                            'SL1_map_depth': sl_info.o1_depth,
                            'SL2_map_depth': sl_info.o2_depth,
                            }
                            sl_info.generate_ds9_regionfiles()

                            with open(diag_name+'.pkl', 'wb') as fp:
                                pickle.dump(sl_diag_data, fp)
                            log.append(str(datetime.datetime.now())+' : '+'diagnostics saved.')

                        except Exception as e: 
                            log.append(str(datetime.datetime.now())+' : '+'diagnostics failed. Reason: '+str(e))
                            pass

                    if not has_ll: log.append(str(datetime.datetime.now())+' : '+'no LL in this AORKEY')

                    if has_ll:
                        log.append(str(datetime.datetime.now())+' : '+'making LL cubes...')
                        try:

                            inputs['channel']=2
                            ll_info = simlacube(inputs)
                            log.append(str(datetime.datetime.now())+' : '+'successfully made LL cubes in '+str(round(ll_info.cube_time,2))+'s')

                        except Exception as e: 
                            log.append(str(datetime.datetime.now())+' : '+'ERROR: '+str(e))
                            failed_cubes.append(str(aor)+' LL')
                            pass


                        try:

                            log.append(str(datetime.datetime.now())+' : '+'now preparing the diagnostics...')
                            diag_name = new_rootdir+run_name+'diagnostics/'+o+'_'+str(aor)+'_'+'_deltat='+str(delta_t)+'_J1='+\
                                                    str(J1_cut)+'_J2='+str(J2_cut)+'_zodicut='+str(zodi_cut)+'_LL_diag'
                            ll_info.generate_report(diag_name+'.pdf')
                            ll_diag_data = {
                            'AORKEY': aor,
                            'cube_time': round(ll_info.cube_time,2),
                            'object': ll_info.object_name,
                            'galactic_coords': ll_info.gal_coords,
                            'zodi_estimate': ll_info.LL1_C_zodi,
                            'mjd': ll_info.AOR_MJD,
                            'number_of_BCDs': ll_info.bcds_in_AOR,
                            'J1_fraction': ll_info.j1_fraction,
                            'J2_fraction': ll_info.j2_fraction,
                            'judge_agreement': ll_info.judge_agreement,
                            'LL1_RMS': ll_info.cube_rms[0],
                            'LL2_RMS': ll_info.cube_rms[1],
                            'LL1_MAD': ll_info.cube_mad[0],
                            'LL2_MAD': ll_info.cube_mad[1],
                            'RAMPTIME': ll_info.RAMPTIME,
                            'SAMPTIME': ll_info.SAMPTIME,
                            'EXPTOT_T': ll_info.EXPTOT_T,
                            'CHNLNUM': 2,
                            'LL1_map_depth': ll_info.o1_depth,
                            'LL2_map_depth': ll_info.o2_depth,
                            }
                            ll_info.generate_ds9_regionfiles()

                            with open(diag_name+'.pkl', 'wb') as fp:
                                pickle.dump(ll_diag_data, fp)
                            log.append(str(datetime.datetime.now())+' : '+'diagnostics saved.')

                        except Exception as e: 
                            log.append(str(datetime.datetime.now())+' : '+'diagnostics failed. Reason: '+str(e))
                            pass


                    with open(log_fname, 'a') as log_f:
                        for l in log:
                               log_f.write(l+' \n')

                    with open(failed_cubes_fname, 'a') as fc_f:
                        for l in failed_cubes:
                               fc_f.write(l+' \n')
                            
    def ds9_to_poly(regfile):
        regdata = ast.literal_eval(open(regfile).read().split('polygon')[-1])
        coords = np.reshape(regdata, (int(len(regdata)/2), 2))
        return coords

    def get_gold_and_simla_specs(gold_cubefile, kind):

        gold_cube = tools.load_cube(gold_cubefile)

        progid = gold_cube.header['PROGID']
        objname = gold_cube.header['OBJECT']
        aorkey = gold_cube.header['AORKEY']
        ending = gold_cubefile.split('cube_gold')[-1].split('_')[2]
        simlacubefile = [path for path,_,_ in os.walk(new_rootdir+run_name) if "PROGID"+str(progid) in path][0]+'/'+\
                                                  objname+'/'+str(aorkey)+'/'+\
                                                  str(aorkey)+'_'+objname+'_'+ending
        simlacube = tools.load_cube(simlacubefile)

        regfile = '/home/work/simla/gold_regions/'+str(progid)+'/'+objname+'_'+ending[0:3]+'_'+kind+'.reg'
        region = ds9_to_poly(regfile)

        l, gold_spec = tools.spectral_extraction(region, gold_cube)
        l, simla_spec = tools.spectral_extraction(region, simlacube)

        return l, gold_spec, simla_spec

    all_goldcubes = []
    golds_rootdir = '/home/work/simla/test_cubes/gold_standard/fitsfiles/'
    for directory in os.listdir(golds_rootdir):
        for obj in os.listdir(golds_rootdir+directory):
            cubes = glob.glob(golds_rootdir+directory+'/'+obj+'/*cube_gold*.fits')
            for i in cubes:
                if 'SL' in i or 'LL' in i:
                    if 'unc' not in i:
                        all_goldcubes.append(i)

    directory = new_rootdir+run_name+'diagnostics/'

    if not os.path.exists(directory+'simlagold_spectra'):
        os.system('mkdir '+directory+'simlagold_spectra')

    sl1_lam = None
    sl2_lam = None
    ll1_lam = None
    ll2_lam = None

    for gold_cubefile in all_goldcubes:

        order = gold_cubefile.split('cube_gold')[-1].split('_')[2][0:3]
        objname = gold_cubefile.split('cube_gold')[-1][1:-9]

        try:
        # if True:

            l, faint_gold, faint_simla = \
                get_gold_and_simla_specs(gold_cubefile, 'faint')

            l, bright_gold, bright_simla = \
                get_gold_and_simla_specs(gold_cubefile, 'bright')

            if order == 'SL1':
                sl1_lam = l
                np.savetxt(directory+'simlagold_spectra/'+objname+'_SL1_SIMLA_faint.txt', faint_simla)
                np.savetxt(directory+'simlagold_spectra/'+objname+'_SL1_GOLD_faint.txt', faint_gold)
                np.savetxt(directory+'simlagold_spectra/'+objname+'_SL1_SIMLA_bright.txt', bright_simla)
                np.savetxt(directory+'simlagold_spectra/'+objname+'_SL1_GOLD_bright.txt', bright_gold)

            elif order == 'SL2':
                sl2_lam = l
                np.savetxt(directory+'simlagold_spectra/'+objname+'_SL2_SIMLA_faint.txt', faint_simla)
                np.savetxt(directory+'simlagold_spectra/'+objname+'_SL2_GOLD_faint.txt', faint_gold)
                np.savetxt(directory+'simlagold_spectra/'+objname+'_SL2_SIMLA_bright.txt', bright_simla)
                np.savetxt(directory+'simlagold_spectra/'+objname+'_SL2_GOLD_bright.txt', bright_gold)

            elif order == 'LL1':
                ll1_lam = l
                np.savetxt(directory+'simlagold_spectra/'+objname+'_LL1_SIMLA_faint.txt', faint_simla)
                np.savetxt(directory+'simlagold_spectra/'+objname+'_LL1_GOLD_faint.txt', faint_gold)
                np.savetxt(directory+'simlagold_spectra/'+objname+'_LL1_SIMLA_bright.txt', bright_simla)
                np.savetxt(directory+'simlagold_spectra/'+objname+'_LL1_GOLD_bright.txt', bright_gold)

            elif order == 'LL2':
                ll2_lam = l
                np.savetxt(directory+'simlagold_spectra/'+objname+'_LL2_SIMLA_faint.txt', faint_simla)
                np.savetxt(directory+'simlagold_spectra/'+objname+'_LL2_GOLD_faint.txt', faint_gold)
                np.savetxt(directory+'simlagold_spectra/'+objname+'_LL2_SIMLA_bright.txt', bright_simla)
                np.savetxt(directory+'simlagold_spectra/'+objname+'_LL2_GOLD_bright.txt', bright_gold)

        except Exception as e: pass

    np.savetxt(directory+'simlagold_spectra/SL1_wavs.txt', sl1_lam)
    np.savetxt(directory+'simlagold_spectra/SL2_wavs.txt', sl2_lam)
    np.savetxt(directory+'simlagold_spectra/LL1_wavs.txt', ll1_lam)
    np.savetxt(directory+'simlagold_spectra/LL2_wavs.txt', ll2_lam)

# test_run = gold_run(0.42, 0.5, 3, None, True, True, False)

# Custom Runs
standard_run = gold_run(3, 0.5, 3, None, True, True, False)
no_inAOR_run = gold_run(3, 0.5, 3, None, True, True, True)
pessimistic_run = gold_run(5, 2, 5, None, False, False, True)
pessimistic_run2 = gold_run(0.1, 0.5, 5, 10, True, True, True)
generous_backgrounds_run = gold_run(3, 2, 5, None, True, True, False)
no_superdark_run = gold_run(3, 0.5, 3, None, True, False, False)
no_superdark_withzodicut_run = gold_run(3, 0.5, 3, 10, True, False, False)
superdark_withzodicut_run = gold_run(3, 0.5, 3, 10, True, True, False)

# delta_t sweep
for dt in [0.1, 0.5, 1, 2, 5, 10, 20]:
    run = gold_run(dt, 0.5, 3, None, True, True, False)
    
# J2 sweep
for J in [0.5, 1, 1.5, 2, 2.5]:
    run = gold_run(3, 0.5, J, None, True, True, False)

# Zodicut sweep
for zc in [1, 3, 5, 10, 15, 20]:
    run = gold_run(3, 0.5, 3, zc, True, True, True)

# INPUTS KEY:
# delta_t, J1_cut, J2_cut, zodi_cut, io_correct, superdark, in_aor
