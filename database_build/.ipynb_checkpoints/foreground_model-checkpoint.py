from mysql.connector import connect
from tqdm import tqdm
import numpy as np

from simla_variables import SimlaVar

connection = connect(host="localhost",user="root",database="SIMLA")
cursor = connection.cursor()

datafile = SimlaVar().simlapath + 'storage/bg_foreground_data.npy'
data = np.load(datafile).T

for aorkey, ism_12, zodi_12 in tqdm(data):

    ADD = f"""
        INSERT INTO foreground_model (AORKEY, ISM_12, ZODI_12) 
        VALUES ({aorkey}, {ism_12}, {zodi_12});
        """
    
    cursor.execute(ADD)
    connection.commit()
    