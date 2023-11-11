# python3
"""
    Transfer JSON data into a SqLite database
    The json data contains the manual mapping data from the manumap gui
    the manumap tool is now changed to process database data directly
"""

from config import get_config
cfg = get_config()
from splib.cute_dialog import start_dialog
import splib.project_db as pdb
import json
import os


class dialog:
    recording = "hus9h"


    layout = """
    title   Transfer manumap JSON data to the database
    text    recording  recording id, example: 'hus1h'
    """

def main():
    if not start_dialog(__file__, dialog):
        return

    # read the json file
    recd = dialog.recording
    json_fn = cfg.work / recd / "manumap.json"

    if not os.path.exists(json_fn):
        print(f"load {recd} failed")
        return

    print("read json blocks")
    with open(json_fn, mode='r') as fi:
        json_data = json.load(fi)

    cbmap = json_data['mappings']

    # prepare database access

    with pdb.DB_Manager() as dbman:
        dbref = dbman.connect(recd, "ml01")
        conn = dbref.conn

        sql_ins = f"""INSERT INTO 'manumap'
                    (cbkey, msoffs, label, lndx)
                    VALUES (?,?,?,?)"""
        sql_del = f"DELETE FROM 'manumap' WHERE cbkey = ?"

        # read mappings for each block
        for cbkey in sorted(cbmap.keys()):
            print(cbkey)

            conn.execute(sql_del, (cbkey,))  # delete data for one block

            # insert all mappings
            for lndx, label, mspos in cbmap[cbkey]:

                conn.execute(sql_ins, (cbkey, mspos, label, lndx))




main()