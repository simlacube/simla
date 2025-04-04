import numpy as np
from astropy.io import fits
from tqdm import tqdm
from mysql.connector import connect

from simla_utils import shard_corners
from simla_variables import SimlaVar
from simladb import query, DB_bcd

irspath = SimlaVar().irspath
connection = connect(host="localhost",user="root",database="SIMLA")
cursor = connection.cursor()

# Set up a library of shard characteristics
# These are structured so that CHNLNUM == list index
edgetrim_lib = [SimlaVar().sl_edgetrim, None, SimlaVar().ll_edgetrim]
nshard_lib = [SimlaVar().sl_n_shards, None, SimlaVar().ll_n_shards]

# Get all of the lowres BCDs, HiRes don't get shards
all_bcd_data = query(DB_bcd.select(DB_bcd.FILE_NAME, DB_bcd.CHNLNUM, DB_bcd.DCEID)\
                          .where((DB_bcd.CHNLNUM==0)|(DB_bcd.CHNLNUM==2)))
all_bcd_filenames = all_bcd_data['FILE_NAME'].to_numpy()
all_bcd_chnlnums = all_bcd_data['CHNLNUM'].to_numpy()
all_bcd_dceids = all_bcd_data['DCEID'].to_numpy()

# Loop through, calculate the shard sky corners, and insert into database
for i in tqdm(range(len(all_bcd_filenames))):
    
    bcdfile = irspath+all_bcd_filenames[i]
    chnlnum = all_bcd_chnlnums[i]
    dceid = all_bcd_dceids[i]
    rectangles, suborders, shard_ids = \
            shard_corners(bcdfile, edgetrim_lib[chnlnum], nshard_lib[chnlnum])

    for j in range(len(rectangles)):
        ras, decs = np.asarray(rectangles[j]).T
        suborder = suborders[j]
        shard_id = shard_ids[j]

        ADD = f"""
            INSERT INTO shard_positions 
            (DCEID, CHNLNUM, SUBORDER, SHARD,
            C0_R, C1_R, C2_R, C3_R,
            C0_D, C1_D, C2_D, C3_D) 
            VALUES 
            ({dceid}, {chnlnum}, {suborder}, {shard_id},
            {ras[0]}, {ras[1]}, {ras[2]}, {ras[3]},
            {decs[0]}, {decs[1]}, {decs[2]}, {decs[3]});
            """
        
        cursor.execute(ADD)
        connection.commit()

        

        
    
    














