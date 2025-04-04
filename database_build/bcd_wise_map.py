import numpy as np
from astropy.io import fits
import glob
from tqdm import tqdm
from mysql.connector import connect

from simla_utils import wise_filename_to_coords, angular_separation
from simla_variables import SimlaVar
from simladb import query, DB_bcd

wisepath = SimlaVar().wisepath

# Collect all WISE files
all_wise_files = glob.glob(wisepath+'**/**/**/*.fits.gz')

# Get the coords of WISE files based on their file name
all_wfile_coords = [wise_filename_to_coords(wfile) for wfile in all_wise_files]
all_wfile_coords = np.asarray(all_wfile_coords)

# Get the FOV coords for all BCDs from the database
q = query(DB_bcd.select(DB_bcd.DCEID, DB_bcd.RA_FOV, DB_bcd.DEC_FOV))
all_bcd_dceids = q['DCEID'].to_numpy()
all_bcd_coords = np.asarray([q['RA_FOV'].to_numpy(), q['DEC_FOV'].to_numpy()]).T

# For each BCD, select the WISE file with the smallest angular separation
bestmatch_wfiles = []
for i in tqdm(range(len(all_bcd_dceids))):
    this_coord = all_bcd_coords[i]
    big_coordarray = np.ones_like(all_wfile_coords)*this_coord
    wise_distances = angular_separation(big_coordarray.T, all_wfile_coords.T)
    best_match = all_wise_files[np.where(wise_distances==np.min(wise_distances))[0][0]]
    bestmatch_wfiles.append(best_match)

# Standardize the names of the WISE files
bestmatch_wfiles = [i.split(wisepath)[-1] for i in bestmatch_wfiles]

# Insert into the database
connection = connect(host="localhost",user="root",database="SIMLA")
cursor = connection.cursor()
for i in tqdm(range(len(all_bcd_dceids))):

    dceid = all_bcd_dceids[i]
    wisefile = bestmatch_wfiles[i]

    ADD = f"""
        INSERT INTO bcd_wise_map 
        (DCEID, WISE_FILE) 
        VALUES 
        ({dceid}, '{wisefile}');
        """
    
    cursor.execute(ADD)
    connection.commit()













    