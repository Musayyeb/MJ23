# python3
'''
    manually mapped labels were done befor the final adjustmenst of block boundaries
    the file workfiles/adjust_block.txt contains the offsets, which we must apply to the mapping
    the numbers in parentesis give the milliseconds to subtract from the map positions
'''
from config import get_config
cfg = get_config()
from splib.cute_dialog import start_dialog
import splib.project_db as pdb

class G:
    pass

class dialog:
    recording = "hus1h"
    chapter = 114
    block = 1

    layout = """
    title   Adjust mapping positions
    label   ----------------------------------------------------
    label   Backup your database before running this !!!
    label   ----------------------------------------------------
    text    recording  recording id, example: 'hus1h'
    """

def main():
    if not start_dialog(__file__, dialog):
        return
    G.recd = dialog.recording
    pdb.db_connector(db_worker)

def db_worker(dbman, vdict=None):
    # do the database processing
    dbref = dbman.connect(dialog.recording, 'ml01')
    conn = dbref.conn

    for cbkey, shift in get_adjust_text():
        print(cbkey, shift)
        sql = "SELECT id, msoffs FROM manumap WHERE cbkey == ?"
        sql_upd = "UPDATE manumap SET msoffs = ? WHERE id = ?"
        csr = conn.execute(sql, (cbkey,))
        for row in csr:
            id, msoffs = row
            newoffs = msoffs - shift
            print(f"update {cbkey}, {id} from {msoffs} to {newoffs}")
            conn.execute(sql_upd, (newoffs, id))


        pass


def get_adjust_text():
    fn = cfg.work / G.recd / 'adjust_blocks.txt'
    with open(fn, mode='r') as fi:
        for line in fi:
            line = line.split('#')[0].strip()
            if line == '':
                continue
            cbkey, shift = line.split()
            shift = shift.replace('(', '').replace(')', '')
            shift = 0 if shift == '' else int(shift)
            if shift == 0:
                continue
            yield cbkey, shift

main()
