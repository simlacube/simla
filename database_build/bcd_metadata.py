from mysql.connector import connect
from tqdm import tqdm
from astropy.io import fits
import glob

from simla_variables import SimlaVar

connection = connect(host="localhost",user="root",database="SIMLA")
cursor = connection.cursor()

irspath = SimlaVar().irspath

all_bcds = glob.glob(irspath+'**/**/**/**/*bcd.fits')

for i in tqdm(all_bcds):

    FILE_NAME = i.split(irspath)[-1]
    
    head = fits.getheader(i)

    if head['AOT_TYPE'] == 'IrsMap':
        stepspar, stepsper = head['STEPSPAR'], head['STEPSPER']
    else:
        stepspar, stepsper = 0, 0

    ADD = f"""
        INSERT INTO bcd_metadata (
        DCEID, FILE_NAME, AORKEY, CHNLNUM, CAMPAIGN, PROGID, OBJECT,
        OBJTYPE, RA_FOV, DEC_FOV, PA_FOV, MJD_OBS, RAMPTIME, SAMPTIME,
        AOT_TYPE, FOVNAME, FOVID, STEPSPAR, STEPSPER) 
        VALUES ({head['DCEID']}, "{FILE_NAME}", {head['AORKEY']}, 
        {head['CHNLNUM']}, "{head['CAMPAIGN']}", {head['PROGID']}, 
        "{head['OBJECT']}", "{head['OBJTYPE']}", {head['RA_FOV']}, 
        {head['DEC_FOV']}, {head['PA_FOV']}, {head['MJD_OBS']}, 
        {head['RAMPTIME']}, {head['SAMPTIME']}, "{head['AOT_TYPE']}", 
        "{head['FOVNAME']}", {head['FOVID']}, {stepspar}, 
        {stepsper});
        """
    
    cursor.execute(ADD)
    connection.commit()