import numpy as np
from astropy.io import fits
from tqdm import tqdm
from mysql.connector import connect
from multiprocessing import Pool
import os

from simla_variables import SimlaVar
from simladb import query, DB_bcd, DB_bcdwise, DB_shardpos, setup_judge1, scorners
from simla_utils import make_wise_bg, DN_to_MJypsr, photometry, fmt_scorners

### MULTIPROCESSING VERSION ###

def get_photometry_and_enter(wfile):
    
    wfile_full = wisepath+wfile

    # Create and subtract the simple background
    image_data, header = fits.getdata(wfile_full), fits.getheader(wfile_full)
    wise_bg = make_wise_bg(image_data)
    bgsub_data = image_data - wise_bg

    # Convert the image to MJy/sr
    usable_wise = DN_to_MJypsr(bgsub_data, 3)

    # Get all shard corners associated with this tile
    sharddata = query(setup_judge1.select( \
            DB_shardpos.DCEID, DB_shardpos.CHNLNUM, DB_shardpos.SUBORDER, DB_shardpos.SHARD, *scorners) \
        .where(((DB_bcd.CHNLNUM==0)|(DB_bcd.CHNLNUM==2)) & (DB_bcdwise.WISE_FILE==wfile)))
    
    s_dceid, s_chnl, s_sub, s_id = \
            sharddata['DCEID'], sharddata['CHNLNUM'], sharddata['SUBORDER'], sharddata['SHARD']
    s_corners = fmt_scorners(sharddata)

    # For each shard, get the photometry and upload to the database
    for i in range(len(s_dceid)):

        dceid, chnl, sub, shardid = s_dceid[i], s_chnl[i], s_sub[i], s_id[i]
        corners = s_corners[i]

        backsub_phot = photometry(corners, usable_wise, header)

        ADD = f"""
            INSERT INTO judge1 (DCEID, CHNLNUM, SUBORDER, SHARD, BACKSUB_PHOT) 
            VALUES ({dceid}, {chnl}, {sub}, {shardid}, {backsub_phot});
            """
        
        cursor.execute(ADD)
        connection.commit()

wisepath = SimlaVar().wisepath
connection = connect(host="localhost",user="root",database="SIMLA")
cursor = connection.cursor()

# Get only the WISE files that are touched by SL or LL
unique_irstarget_wise = np.unique(query( \
    setup_judge1.select(DB_bcdwise.WISE_FILE).where((DB_bcd.CHNLNUM==0)|(DB_bcd.CHNLNUM==2))) \
                                  ['WISE_FILE'].to_numpy())

if __name__ == '__main__':
    with Pool(processes=8) as pool: 
        for _ in tqdm(pool.imap_unordered(get_photometry_and_enter, unique_irstarget_wise), \
                           total=len(unique_irstarget_wise)):
            pass

# Note: the progress bar was one tick away from finish, and it is still uploading.
# do not quit it, it's not stuck.

    




    