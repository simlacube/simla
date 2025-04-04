from sqlalchemy import create_engine, text
from pypika import Schema, Query as pq
import pypika
import pandas as pd
import pymysql

from simla_variables import SimlaVar

# Aliases for database tables
DB_bcd = pypika.Table("bcd_metadata")
DB_bcdwise = pypika.Table("bcd_wise_map")
DB_shardpos = pypika.Table("shard_positions")
DB_judge1 = pypika.Table("judge1")
DB_judge2 = pypika.Table("judge2")
DB_foreground = pypika.Table("foreground_model")

# Convenience calls
scorners = [DB_shardpos.C0_R, DB_shardpos.C1_R, DB_shardpos.C2_R, DB_shardpos.C3_R,
            DB_shardpos.C0_D, DB_shardpos.C1_D, DB_shardpos.C2_D, DB_shardpos.C3_D]
j2spec = [DB_judge2.SPEC0, DB_judge2.SPEC1, DB_judge2.SPEC2, DB_judge2.SPEC3, DB_judge2.SPEC4, 
          DB_judge2.SPEC5, DB_judge2.SPEC6, DB_judge2.SPEC7, DB_judge2.SPEC8, DB_judge2.SPEC9]

# Join the tables together for convenience

# Use these while building the database, since
# you might not have all tables at each step
# Do not exclude anything while the DB is being built!
setup_judge1 = pq.from_(DB_bcd)\
    .join(DB_bcdwise).using('DCEID')\
    .join(DB_shardpos).using('DCEID')

setup_superdark = pq.from_(DB_bcd)\
    .join(DB_judge1).using('DCEID')\
    .join(DB_foreground).using('AORKEY')\

# Final Database
simladb = pq.from_(DB_bcd)\
    .join(DB_bcdwise).using('DCEID')\
    .join(DB_shardpos).using('DCEID')\
    .join(DB_judge1).using('DCEID', 'SUBORDER', 'SHARD')\
    .join(DB_judge2).using('DCEID', 'SUBORDER', 'SHARD')\
    .join(DB_foreground).using('AORKEY')
# must join using all shard parameters to avoid duplicate rows
# CHNLNUM is left out since it is also in bcd_metadata, 
# but each DCEID has only one CHNLNUM so it works

# Final DB but excluding unwanted data
simladbX = simladb.where(\
           (DB_bcd.OBJTYPE.notin(SimlaVar().banned_objtypes))&\
           (DB_bcd.OBJECT.notin(SimlaVar().banned_objects))&\
           (DB_judge1.BACKSUB_PHOT!=0.0)&\
           ((DB_bcd.CHNLNUM==0)|(DB_bcd.CHNLNUM==2)))

# Each time an alias is made, add it to this list too
tables = [DB_bcd, DB_bcdwise, DB_shardpos, DB_judge1, DB_judge2, DB_foreground]

# Info for connecting to DB (might need to be changed per machine)
host = 'localhost'
database = 'SIMLA'
engine = create_engine("mysql+pymysql://root@localhost/SIMLA")

# The function for connecting to and querying the database
def query(Command):
    with engine.connect() as connection:
        record = connection.execute(text("SELECT DATABASE();")).fetchone()
        tb=Schema('INFORMATION_SCHEMA').COLUMNS
        base=pq.from_(tb).select(tb.COLUMN_NAME)
        for t in tables:
            connection.execute(text(str(base.where(tb.TABLE_NAME==t.get_table_name()))))
            data = pd.read_sql(str(Command), con=connection)
    return data
