# python3
'''

    there are json files with the result of the mapping (map_nbr.py)

    evaluate the result to find sequences of letters, which we can assume
    to be mapped correct.

    put these letter into a new list, then

        1.  try to identify syllables, which we put into the new database
        2.  identify letters, which we could put back into the training
'''


from splib.cute_dialog import start_dialog
import sys


class dialog:
    recording = "hus9h"
    chapno = ''
    blkno = ''

    layout = """
    title Nullmap - start mapping
    text     recording  recording id, example: 'hus1h'
    text     chapno    Chapter or a list of chapter n:m
    text     blkno     Block or list of blocks 3:8 or '*'
    """
if not start_dialog(__file__, dialog):
    sys.exit(0)


from config import get_config, AD
cfg = get_config()

import splib.toolbox as tbx
import splib.attrib_tools as att
import splib.text_tools as tt
from machine_learning.ml_tools import PredLoader
from dataclasses import dataclass
import json
import os

class G:
    pass

vowels = 'aiuAYNW*'

@dataclass
class MappedLetter:
    ltr : str
    lndx: int  # letter index
    repeat: int  # ltr repetition
    lpos : int  # start position (ms)
    rpos : int  # end position
    ratg : float  # rating


def main():

    recd, chap_no, blk_no = dialog.recording, dialog.chapno, dialog.blkno

    G.blk_counter = tbx.get_block_counters(recd)
    if ':' in chap_no:  # chap_no is either a single number or a range of n:m
        chp1, chp2 = chap_no.split(':')
        chp_range = range(int(chp1), int(chp2))
    else:
        chp_range = range(int(chap_no), int(chap_no)+1)

    for chapno in chp_range:

        if blk_no == '*':
            blk_range = range(1, G.blk_counter[chapno]+1)
        elif ':' in blk_no:  # blockno is either a single number or a range of n:m
            blk1, blk2 = blk_no.split(':')
            blk_range = range(int(blk1), int(blk2))
        else:
            blk_range = range(int(blk_no), int(blk_no)+1)

       # this program processes n blocks in sequence
        for blkno in blk_range:

            json_fn = cfg.data / recd / 'mapping_results' /f"pred_results {chapno:03d}.{blkno:03d}.json"
            if not os.path.exists(json_fn):
                continue

            print("found file:", json_fn)
            with open(json_fn, mode='r') as fi:
                results = json.load(fi)

            # results is a list of pairs:
            # for each iteration and each mode, there is one pair
            # each pair is a block info and the mapping table

            vect_loader = att.VectLoader(recd, chapno, blkno)
            xdim = vect_loader.get_xdim()  # duration of block determins the x dimension of chart
            print(f"got results: len={len(results)} blkinfo={results[0]}")

            origin, mapped = convert_results(results)

            rated = judge_maps(origin, mapped)
            print("rated:")
            for item in rated:
                ltr, lndx, rpt, ct, lmin, lmax, rmin, rmax = item
                minlen, maxlen = rmin - lmax, rmax - lmin
                pos = int((lmax+rmin)/2)
                print(f"{ltr} ({rpt}) lx:{lndx:3d} ct:{ct:2d}  pos:{pos:5d}  len:{minlen:4}:{maxlen:4}")


def judge_maps(origin, mapped):
    rated = []
    for key, llist in sorted(mapped.items()):

        lmax = rmax = mcount = 0
        lmin = rmin = 99999
        for orig, map in zip(origin, llist):
            if map.rpos > 0:
                mcount += 1
                lmin = min(lmin, map.lpos)
                lmax = max(lmax, map.lpos)
                rmin = min(rmin, map.rpos)
                rmax = max(rmax, map.rpos)
        rated.append((map.ltr, map.lndx, map.repeat, mcount, lmin, lmax, rmin, rmax))

    return rated



def convert_results(results):
    # put together all results for a specific letter from the different mappings
    mapped = {}  # key = (lndx, ltr) data = list of AD()s
    origin = []  # list of AD()s: {chap blk iter mode ratg txtlen mslen} - list index = orx

    for blk_info, mappings in results:
        origin.append(AD(blk_info))

        for ltr_attr in mappings:
            ltr = ltr_attr['ltr']
            lndx = ltr_attr['lndx']
            key = (lndx, ltr)
            if not key in mapped:
                mapped[key] = []
            mapped[key].append(AD(ltr_attr))

    print("origin tab:")
    print(origin)
    print()
    print("mapped tab:")
    print(mapped)

    return origin, mapped



main()