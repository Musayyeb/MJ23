# python 3
'''
    Prepare training data

    Collect audio attributes from stored numpy files

    The numpy files contain vectors of audio attributes, which were
    calculated from different tools. The attributes are just numbers
    The vectors provide the attribute values of the audio source in
    5 millisecond intervals.

    Training data is prepared for manually mapped letter (label)
    to time positions. The manumap data is stored in the
    "ml01" database, "manumap" table.

    The training data table is created here. The training data is a
    table with fields for all available attributes. For each mapping
    spot, one record is stored, with all attributes.

    We store additional records, for the attributes in +-5
    and +-10 millisecond distance from the originally mapped spot,
    so each manumap spot is represented by 5 attribute records.


'''
from splib.cute_dialog import start_dialog
import sys

class dialog:
    recording = "hus1h"
    tbname = ''
    chapter = 114
    block = 1

    layout = """
    title   prepare training data
    label   now the training data is also prepared for the automap data
    text    tbname     is 'manumap' or 'automap'      
    text    recording  recording id, example: 'hus1h'
    text    chapter    koran chapter number (or '*')
    text    block      block number (or '*')
    """

if not start_dialog(__file__, dialog):
    sys.exit(0)



from config import get_config
cfg = get_config()
import numpy as np
import splib.attrib_tools as at
import splib.project_db as pdb
import splib.attrib_tools as at

class G:
    recd = ''

def main():
    G.recd = dialog.recording
    pdb.db_connector(db_worker)

def db_worker(dbman, vdict=None):
    # do the database processing
    tbname = dialog.tbname
    dbname = 'ml01' if tbname == 'manumap' else 'ml02'
    dbref = dbman.connect(dialog.recording, dbname)
    conn = dbref.conn
    spots = training_spots(conn)

    create_train_data(conn, spots)

def create_train_data(conn, spots):
    # retrieve the manumap keys from the list of spots
    # for each key create the training data records
    recd = dialog.recording
    curr_chap = curr_blk = -1

    all_np_vect = None
    namelist = []
    commit_ct = 0
    sql = "REPLACE INTO 'train_data' ({}) VALUES ({})"
    for cbkey, mspos, label in spots:

        #print('row: ', cbkey, mspos, label)
        chapno, blkno =  int(cbkey[0:3]), int(cbkey[4:7])
        if (chapno, blkno) != (curr_chap, curr_blk):
            curr_chap, curr_blk = chapno, blkno

            all_np_vect = at.AllVectors(recd, chapno, blkno)
            names = ['cbkey', 'msoffs', 'label']
            names.extend(all_np_vect.get_names())
            namelist = ', '.join(names)
            # print('np_vect namelist:', namelist)
            commalist = ','.join(["?" for _ in names])

        for offs, fix in ((-10,'--'), (-5,'-'), (0,''), (5,'+'), (10,'++')):
            pos = mspos + offs
            lbl = label + fix
            values = [cbkey, pos, lbl]
            try:
                values.extend(all_np_vect.get_values(pos))
            except IndexError as ex:
                print(f"get values from vector: {values}")
                raise
            sql = sql.format(namelist, commalist)
            # print(sql)
            conn.execute(sql, (values))
        commit_ct += 1
        if commit_ct > 100:
            conn.commit()
            commit_ct = 0
    conn.commit()


def training_spots(conn):
    # this is a generator - yield the manumap spots for the
    # requested chapter / block
    schap, sblk = dialog.chapter, dialog.block

    sql = f"""SELECT cbkey, msoffs, label FROM '{dialog.tbname}'
                ORDER BY cbkey, msoffs
          """
    csr = conn.execute(sql)
    for row in csr:
        cbkey = row[0]
        chapno = int(cbkey[0:3])
        blkno = int(cbkey[4:7])
        if ((schap == '*' or int(schap) == chapno)
            and (sblk == '*' or int(sblk) == blkno)):
            yield row

main()