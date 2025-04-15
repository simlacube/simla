'''
Home for global variables used throughout SIMLA.
Set these before uploading database tables.

'''

class SimlaVar:
    def __init__(self):

        # Paths
        self.simlapath = '' # /path/to/directory/with/ simlacube.py
        self.irspath = '' # /path/to/directory/with/ irsstare
        self.wisepath = '' # /path/to/wise/images/
        self.zodipath = '' # /path/to/zodi/spectra/
        self.runpath = '' # /path/to/store/full/runs/
        self.cubismpath = '' # /path/to/CUBISM/installation/
        # self.bgsub_wisepath = ''

        # Shard characteristics
        self.sl_edgetrim = 8 # % on both sides
        self.ll_edgetrim = 4 # % on both sides
        self.sl_n_shards = 5
        self.ll_n_shards = 5

        # Specifications for superdark creation
        self.n_zodi_bins = 4
        # These are *not* for cubes, but cuts for making superdarks
        self.judge1_sd_cut = (-0.1, 0.1) # MJy/sr
        self.ism_sd_cut = 1 # MJy/sr
        self.sd_trim_sigma = 1.5

        # Use these to exclude certain BCDs
        self.banned_objtypes = ['TargetMovingSingle', 'TargetMovingCluster', 'TargetMulti']
        self.banned_objects = ['NCVZ-dark-spot']
        self.banned_aorkeys = [9107456] # has no uncertainties for some reason

        # Constants
        self.sl_ramptimes = [6.29, 14.68, 241.83, 60.95]
        self.ll_ramptimes = [6.29, 14.68, 31.46, 121.90]
        self.ll_gain_change_mjd = 54403.0

        # Computation
        self.processors = 8

        