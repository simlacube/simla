'''
Take the pre-computed zodi and ISM scalers and upload to the 
foreground table in the SIMLA database.

These scalers are at a specific wavelength (12um) and serve as an 
overall indicator of zodi and ISM strength for an AOR.

The pre-computed data must be in simlapath/storage/bg_foreground_data.npy
No prerequisite code.

'''

from mysql.connector import connect
from tqdm import tqdm
import numpy as np

from simla_variables import SimlaVar

# Establish connection to SIMLA database
connection = connect(host="localhost",user="root",database="SIMLA")
cursor = connection.cursor()

# File containing pre-computed zodi and ISM scalers
datafile = SimlaVar().simlapath + 'storage/bg_foreground_data.npy'
data = np.load(datafile).T

# Upload to the database for each AOR
for aorkey, ism_12, zodi_12 in tqdm(data):

    ADD = f"""
        INSERT INTO foreground_model (AORKEY, ISM_12, ZODI_12) 
        VALUES ({aorkey}, {ism_12}, {zodi_12});
        """
    
    cursor.execute(ADD)
    connection.commit()
    