# python3
'''
    The lmap table is initially created without values for frequency and amplitude
    here the missing values are updated
'''



from splib.cute_dialog import start_dialog
import sys


class dialog:
    recording = "hus1h"
    chapno = ''
    blkno = ''

    layout = """
    title Make phonem DB
    text     recording  recording id, example: 'hus1h'
    text     chapno    Chapter or a list of chapter n:m
    text     blkno     Block or list of blocks 3:8 or '*'

    """
if not start_dialog(__file__, dialog):
    sys.exit(0)


from config import get_config, AD
cfg = get_config()

import splib.toolbox as tbx
import splib.text_tools as tt
import splib.attrib_tools as att
import splib.project_db as pdb
from dataclasses import dataclass
from splib.db_schema import LMAP

class G:
    dbman = None
    conn = None

vowels = 'aiuAYNW*'

def main():
    pdb.db_connector(db_worker)

def db_worker(dbman):
    G.dbman = dbman

    dbref = dbman.connect(dialog.recording, 'proj')
    G.conn = dbref.conn

    sql = """ UPDATE lettermap 
              SET lfreq = ? ,
                  rfreq = ? ,
                  lampl = ? ,
                  rampl = ?
              WHERE id = ?  """

    recd, chap_seq, blk_seq = dialog.recording, dialog.chapno, dialog.blkno

    failcount = 0
    updcount = 0
    for chapno, blkno in tbx.chap_blk_iterator(recd, chap_seq, blk_seq):
        print(f"do {chapno:03d}_{blkno:03d}")
        vect_loader = att.VectLoader(recd, chapno, blkno)
        pyaa, rosa, freq = vect_loader.get_vectors()

        seq = list(read_lettermap_db(recd, chapno, blkno))


        for lmap in seq:
            #print(lmap)
            if lmap.lpos == 0 or lmap.rpos == 0:
                failcount += 1
                continue

            lfreq = int(freq[lmap.lpos])
            rfreq = int(freq[lmap.rpos])
            lampl = int(rosa[lmap.lpos])
            rampl = int(rosa[lmap.rpos])

            G.conn.execute(sql, (lfreq, rfreq, lampl, rampl, lmap.id))
            updcount += 1
        G.conn.commit()

    print(f"updated: {updcount}, failed: {failcount}")

def read_lettermap_db(recd, chapno, blkno):
    cbkey = f"{chapno:03d}_{blkno:03d}"
    sql = "SELECT * FROM lettermap WHERE recd == ? and cbkey == ? ORDER BY lndx"
    csr = G.conn.execute(sql, (recd, cbkey))
    for row in csr.fetchall():
        lmap = LMAP(*row)
        yield lmap

def is_cons(ltr):
    return not ltr in vowels
def is_vow(ltr):
    return ltr in vowels

main()