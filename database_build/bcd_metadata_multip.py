'''
Extract useful info from the headers of IRS BCDs and 
upload these to the bcd_metadata table in the SIMLA database.

Make sure irspath is set in simla_variables.py
No prerequisite code.

This is the version with multiprocessing enabled. Specify the cores to use in simla_variables.py.

'''

from mysql.connector import connect
from tqdm import tqdm
from astropy.io import fits
import glob
from multiprocessing import Pool

from simla_variables import SimlaVar

# Establish connection to the database
connection = connect(host="localhost",user="root",database="SIMLA")
cursor = connection.cursor()

irspath = SimlaVar().irspath

# This gathers the paths for all BCDs
all_bcds = glob.glob(irspath+'**/**/**/**/*bcd.fits')

def inserter(i):

    # We are only uploading the invariable part of the path, after irspath
    FILE_NAME = i.split(irspath)[-1]
    
    head = fits.getheader(i)

    # Staring mode obs have no mapping parameters, input fake ones
    if head['AOT_TYPE'] == 'IrsMap':
        stepspar, stepsper = head['STEPSPAR'], head['STEPSPER']
    else:
        stepspar, stepsper = 0, 0

    # One target has a ' in the object name, needs to be removed
    objname = head['OBJECT'].replace("'","")

    ADD = f"""
        INSERT INTO bcd_metadata (
        DCEID, FILE_NAME, AORKEY, CHNLNUM, CAMPAIGN, PROGID, OBJECT,
        OBJTYPE, RA_FOV, DEC_FOV, PA_FOV, MJD_OBS, RAMPTIME, SAMPTIME,
        AOT_TYPE, FOVNAME, FOVID, STEPSPAR, STEPSPER) 
        VALUES ({head['DCEID']}, '{FILE_NAME}', {head['AORKEY']}, 
        {head['CHNLNUM']}, '{head['CAMPAIGN']}', {head['PROGID']}, 
        '{objname}', '{head['OBJTYPE']}', {head['RA_FOV']}, 
        {head['DEC_FOV']}, {head['PA_FOV']}, {head['MJD_OBS']}, 
        {head['RAMPTIME']}, {head['SAMPTIME']}, '{head['AOT_TYPE']}', 
        '{head['FOVNAME']}', {head['FOVID']}, {stepspar}, 
        {stepsper});
        """
    
    cursor.execute(ADD)
    connection.commit()

if __name__ == '__main__':
    with Pool(processes=SimlaVar().processors) as pool: 
        for _ in tqdm(pool.imap_unordered(inserter, all_bcds), \
                           total=len(all_bcds)):
            pass
