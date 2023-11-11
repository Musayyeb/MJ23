'''

    mapping data is now in a json file

    the json file has mapping info for the same letter from different mapping algorithms

    Here we check, which letters are safely mapped, and only store to good ones into
    the datebase
'''


from splib.cute_dialog import start_dialog
import sys


class dialog:
    recording = "hus1h"
    chap = ''
    blk = ''
    db_write = True

    layout = """
    title dbwrite json lettermap
    text     recording  recording id, example: 'hus1h'
    text     chap       chapter range
    text     blk        block range
    bool     db_write  write the database - uncheck for testing 
    """
if not start_dialog(__file__, dialog):
    sys.exit(0)


from config import get_config, AD
cfg = get_config()

import splib.toolbox as tbx
import splib.project_db as pdb
import json
import os

class G:
    incount = 0
    outcount = 0

def main():
    pdb.db_connector(db_worker)


def db_worker(dbman):
    dbref = dbman.connect(dialog.recording, 'proj')
    conn = dbref.conn

    ltrmap_json = cfg.work / 'mapped_letters.json'
    if os. path.exists(ltrmap_json):
        ltrmap = json.load(open(ltrmap_json, mode='r'))
    else:
        raise Exception(f"there is no json file {ltrmap_json}")

    print("size of json (bytes):", sys.getsizeof(ltrmap))
    print("size of json (keys):", len(ltrmap))

    keys = sorted(ltrmap.keys())  # this is big!

    if dialog.db_write:
        prev_blk = ()
        prev_rpos = 0
        blk_results = []

        def end_block(chapno, blkno):
            # print("called end_block for",chapno, blkno)
            nonlocal blk_results
            if blk_results:
                write_letterbase(conn, dialog.recording, chapno, blkno, blk_results)
                conn.commit()
                blk_results = []


        for datalist in get_all_keys(keys, ltrmap, end_block):
            G.incount += 1
            best, prev_pos = process(datalist, prev_rpos)

            if best.ratg > -1:
                blk_results.append(best)
                G.outcount += 1

    else:

        for chapno, blkno in tbx.chap_blk_iterator(dialog.recording, dialog.chap, dialog.blk):

            blk_results = []
            prev_rpos = 0

            for datalist in get_selected_keys(chapno, blkno, keys, ltrmap):

                G.incount += 1
                best, prev_pos = process(datalist, prev_rpos)

                if best.ratg > -1:
                    blk_results.append(best)
                    G.outcount += 1

            if dialog.db_write:
                write_letterbase(conn, dialog.recording, chapno, blkno, blk_results)
                conn.commit()

    print("input count", G.incount, "written to db:", G.outcount)

def get_all_keys(json_keys, json_data, cb_endblock):
    keylist = []
    prevblk = (0,0)   # chap, blk
    prevltr = (0,0,0)
    for rkey in json_keys:
        chap, blk, lndx, ltr, algo = rkey.split()
        chapno, blkno = int(chap), int(blk)
        ltrkey = (chapno, blkno, lndx)
        blkkey = (chapno, blkno)
        if blkkey != prevblk:
            cb_endblock(*prevblk)
            prevblk = blkkey
        if ltrkey != prevltr:
            if keylist:
                yield keylist
            keylist = []
            prevltr = ltrkey

        keylist.append((rkey, json_data[rkey]))

    if keylist:
        yield keylist

    cb_endblock(*prevblk)
    return

def get_selected_keys(chapno, blkno, json_keys, json_data):
    keylist = []
    prevkey = ()
    for rkey in json_keys:
        chap, blk, lndx, ltr, algo = rkey.split()
        if int(chap) == chapno and int(blk) == blkno:

            compkey = (chap, blk, lndx)
            if compkey != prevkey:
                if keylist:
                    yield keylist
                keylist = []
                prevkey = compkey

            keylist.append((rkey, json_data[rkey]))
    if keylist:
        yield keylist
    return


def process(datalist, prev_rpos):
    p = not dialog.db_write  # either print or write the database
    # expect one entr for each algo
    if not datalist:
        return AD(ratg=-1), 0
    lenlist = []
    poslist = []
    best_ratg = 0
    best_ndx = -1
    best_key = ''

    for ndx, (k, d) in enumerate(datalist):
        repeat, lpos, rpos, ratg = d
        if ratg > -1:
            length = rpos - lpos
            lenlist.append(length)
            poslist.append(lpos)
        if ratg > best_ratg:
            best_ratg = ratg
            best_ndx = ndx
            best_key = k
    lendiff = max(lenlist) - min(lenlist) if lenlist else -1
    posdiff = max(poslist) - min(poslist) if poslist else -1
    if repeat > 1:
        lendiff = int(lendiff / repeat)
        posdiff = int(posdiff / repeat)

    _, d = datalist[best_ndx]
    repeat, lpos, rpos, ratg = d  # gives us the best lpos and rpos
    if p: print(f"{' '*38} <{lpos-prev_rpos:5d}>")
    for k,d in datalist:
        _, _, _, _, algo = k.split()
        repeat, lpos, rpos, ratg = d
        if p: print(f"{' '*len(k)} {repeat:2d} {lpos:5d} {rpos:5d} ({rpos-lpos:4d}) >{int((lpos+rpos)/2):5d}<  {ratg:5.2f}  {algo}")

    k, d = datalist[best_ndx]
    repeat, lpos, rpos, ratg = d
    chap, blk, lndx, ltr, algo = k.split()
    if p: print(f"{k} {repeat:2d} {lpos:5d} {rpos:5d} ({rpos-lpos:4d}) >{int((lpos+rpos)/2):5d}<  {ratg:5.2f}        dlen {lendiff:5d}   dpos {posdiff:5d}")

    # Decide: which goes into the lettermap database?
    if lendiff > 50 or posdiff > 70:
        return AD(ratg=-1), rpos

    best = AD(lndx=int(lndx), ltr=ltr, repeat=repeat, lpos=lpos, rpos=rpos, ratg=ratg, lgap=0, rgap=0)

    return best, rpos



def write_letterbase(conn, recd, chapno, blkno, results):
    # this is called per block
    print(f"write db {chapno}-{blkno} mappings: {len(results)}")
    cbkey = f"{chapno:03d}_{blkno:03d}"
    sql = 'DELETE FROM lettermap WHERE recd == ? and cbkey == ?'
    conn.execute(sql, (recd, cbkey))

    # at the very ends of the block set the gaps to zero
    results[0].lgap = 0
    results[-1].rgap = 0
    for l1, l2 in zip(results, results[1:]):
        l1.rgap = l2.lgap = l2.lpos - l1.rpos

    sql = """INSERT INTO lettermap (ltr, recd, cbkey, lndx, rept, ratg, 
         lpos, rpos, tlen, lgap, rgap)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

    for adx, ad in enumerate(results):
        if ad.ratg == -1:
            continue

        if adx == 0 and ad.lndx > 0 and results[1].lndx == 0:
            print("skipped", ad)
            continue

        tlen = ad.rpos - ad.lpos
        ratg = round(ad.ratg, 2)
        conn.execute(sql, (ad.ltr, recd, cbkey, ad.lndx, ad.repeat, ratg,
                           ad.lpos, ad.rpos, tlen, ad.lgap, ad.rgap))

    return

main()