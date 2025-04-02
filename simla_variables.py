# Home for global variables

class SimlaVar:
    def __init__(self):

        # Paths
        self.simlapath = '/Users/grant/Desktop/Research/SIMLA/simla/'
        self.irspath = '/Users/grant/Data/spitzer_irs/irs_data/'
        self.wisepath = '/Users/grant/Data/wise/irsa-allwise/'
        # self.bgsub_wisepath = '/Users/grant/Desktop/Research/SIMLA/bgsub_wise/'
        self.zodipath = '/Users/grant/models/zodimodel_irs/zodi_spectra/'

        # Shard characteristics
        self.sl_edgetrim = 8 # % on both sides
        self.ll_edgetrim = 0 # % on both sides
        self.sl_n_shards = 5
        self.ll_n_shards = 5

        # Specifications for superdark creation
        self.n_zodi_bins = 4
        # These are *not* for cubes, but stricter cuts for making superdarks
        self.judge1_sd_cut = (-0.1, 0.1) # MJy/sr
        self.ism_sd_cut = 0.5 # MJy/sr
        self.sd_trim_sigma = 1.5

        # Use these to exclude certain BCDs
        self.banned_objtypes = ['TargetMovingSingle', 'TargetMovingCluster']
        self.banned_objects = ['NCVZ-dark-spot']

        # Constants
        self.sl_ramptimes = [6.29, 14.68, 241.83, 60.95]
        self.ll_ramptimes = [6.29, 14.68, 31.46, 121.90]
        self.ll_gain_change_mjd = 54403.0

        