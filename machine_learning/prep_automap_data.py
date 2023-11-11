# python3
'''
    prepare automap data

    after successfully predict mappings, we want to improve training by extending the training dataset

    Include some of the good results for a deeper training

    the good results are selected according to the high letter ratings
    Select (semi)voiced consonants,  and the vowels 'a', 'i' and 'u'

    The existing mapping data is in project_db/lettermap and contains data from hus1h and hus9h

    the manumap data is in hus9h/mldata_01/manumap
    the automap data goes to hus9h/mldata_02/automap

'''

from splib.cute_dialog import start_dialog
import sys


class dialog:
    recording = ""
    db_write = False
    plot = False

    layout = """
    title prepare automap training data
    text     recording  Recording (data goes into mldata_02)
    bool     db_write  write the database (not for testing)
    bool     plot      plotting will come eventually
    """
if not start_dialog(__file__, dialog):
    sys.exit(0)


from config import get_config, AD
cfg = get_config()

from matplotlib import pyplot as plt
import splib.toolbox as tbx
import splib.project_db as pdb
import splib.attrib_tools as att
from splib.db_schema import LMAP
import itertools, math
import random


class G:
    conn = None
    traindata = None

#all_ltrs = '.abdfhiklmnqrstuwyzġšǧʻʼḍḏḥḫṣṭṯẓ*ALNWY'
selected = 'abdhilmnruwyzʻḏẓ*ALNWY'


def main():
    pdb.db_connector(dbwriter)


def dbwriter(dbman):
    # create the TrainDataWriter object
    recd = dialog.recording
    dbname = "ml02"  # for the automap table
    tbname = 'automap'
    dbref = dbman.connect(recd, dbname)
    conn = dbref.conn
    if dialog.db_write:
        sql = f"DELETE FROM {tbname} where cbkey != ?"
        conn.execute(sql,("moon",))
        conn.commit()
    G.traindata = TrainDataWriter(conn, recd, tbname)

    pdb.db_connector(db_worker)

def db_worker(dbman):

    dbref = dbman.connect('', 'proj')  # no recording for proj
    conn = G.conn = dbref.conn

    count = itertools.count()
    for ltr in selected:
        min_ratg = 25
        fgrps = {}
        for lmap in read_lettermap_db(ltr, min_ratg):
            if lmap.recd != dialog.recording:
                continue # Only pick mappings which match recording

            chapno, blkno = [int(x) for x in lmap.cbkey.split('_')]

            freq = att.get_cached_vector(lmap.recd, chapno, blkno, 'freq')
            #print(type(freq), freq.shape, freq[6000:6020])

            mpos = int((lmap.lpos + lmap.rpos)/2)
            mfreq = int(freq[mpos])
            if 80 < mfreq < 350 or ltr == 'h':
                fg = group_by_freq(mfreq)
                if not fg in fgrps:
                    fgrps[fg] = []
                #print(mpos, mfreq)
                seqno = next(count)
                fgrps[fg].append((mfreq, seqno, lmap))

        for fg, cand in sorted(fgrps.items()):
            #cand.sort()
            # print(f"\n{ltr} {fg} {len(cand)}")

            ct = 0
            k = min(20, len(cand))
            data = random.sample(cand, k=k)
            for mfreq, _, lmap in data:
                G.traindata.put(lmap)
                #print(mfreq, lmap)



def group_by_freq(f):
    # get a numerical index (int) for bands of frequencies (logarithmic)
    if f == 0: return 0
    if f < 80: return 1
    g = int((math.log(f) - 4.0) * 15)
    return g


def read_lettermap_db(ltr, min_ratg):
    sql = "SELECT * FROM lettermap WHERE ltr == ? and ratg >= ?"
    csr = G.conn.execute(sql, (ltr, min_ratg))
    for row in csr.fetchall():
        lmap = LMAP(*row)
        yield lmap


class TrainDataWriter:
    def __init__(self, conn, recd, tbname):
        self.conn = conn
        self.recd = recd
        self.tbname = tbname
        self.limit = 0


    def put(self, lmap):
        if dialog.db_write:
            sql = f"""INSERT INTO {self.tbname} (cbkey, msoffs, label, lndx)
                     VALUES (?, ?, ?, ?)"""
            msoffs = int((lmap.lpos + lmap.rpos)/2)
            self.conn.execute(sql, (lmap.cbkey, msoffs, lmap.ltr, lmap.lndx))
            self.limit += 1
            if self.limit > 100:
                print("commit")
                self.conn.commit()  # commit each time ???
                self.limit = 0

main()