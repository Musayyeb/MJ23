                                                                                                                                                                                                                                                  # python3
'''

    This mapping approach is based on map_nbr.py

    Instead of working along each of the iterations (different trainings of the same model)
    by itself (with the risk of getting out of syunc)
    the mapping algorithm searches each letters in all of the iterations. This allows to
    pick the "best" letter, which makes is safer to determine tshe posiotion of the next letters

    The desicion, which iteration returned the best letter may be tricky - but it SHOULD work.

    Create one artificial rating, where "friendly letters" add up their probabilities. These "supported"
    predictions are helpful in a few cases.
'''


from splib.cute_dialog import start_dialog
import sys


class dialog:
    recording = "hus1h"
    chapno = ''
    blkno = ''
    name = ''
    iters = '0'
    debug = False
    usepnp = False
    use_support = True
    db_write = False
    avgwin = 20
    algo = ''

    layout = """
    title map_multi - while mapping, observe multiple iterations
    text     recording  recording id, example: 'hus1h'
    text     chapno    Chapter or a list of chapter n:m
    text     blkno     Block or list of blocks 3:8 or '*'
    text     name      model name
    text     iters     one or more specific iterations
    bool     usepnp    use PNP for curpos
    bool     debug     show extra messages about predictions and ratings
    bool     use_support  duplicate predictions for nbr support
    label    the code does not overwrite existing database entries
    bool     db_write  write the database - uncheck for testing 
    text     algo      algorithm for current position ('mj', 'hb', ...)
    int      avgwin   window size for average probability (slices)
    """
if not start_dialog(__file__, dialog):
    sys.exit(0)


from config import get_config, AD
cfg = get_config()

import numpy as np
import splib.toolbox as tbx
import splib.sound_tools as st
from itertools import product
import splib.attrib_tools as att
import splib.text_tools as tt
import splib.project_db as pdb
from machine_learning.ml_tools import PredLoader, get_training_specs
from scipy.stats import norm
from dataclasses import dataclass, field
from collections import Counter
from typing import List
import json
import codecs
import random
import time
import os
import traceback

from matplotlib import pyplot as plt

class G:
    ml_result = None  # json writing oject
    lc = att.LabelCategories
    colors = None  # colormix stored here
    mapped_letters = []
    results = []  # ratings for blocks and iterations
    friends = {}
    support_list = []
    all_data = {}  # collection of block related results
    debug = True  # trigger print statements
    statfile = ""  # statistics file name

vowels = 'aiuAYNW*'

# all ltrs: '.abdfhiklmnoqrstuwyzġšǧʻʼḍḏḥḫṣṭṯẓ*ALNWY'
support_weights = {0:3, 1:5, 2:8, 3:11, 4:15, 5:20, 6:30, 7:40, 8:50, 9:60}
friends = [ # the x-axis supports the y-axis
'''
_ b d k q r t ǧ ṭ l L ġ 
b _ 4 . 4 . . 4 4 . . 4
d 4 _ . 4 . 4 4 4 . . 4
k . . _ 3 . . . . . . 3
q 4 4 4 _ . . 4 4 . . 5
r . . . . _ . . . 4 2 .
t . 4 . . . _ . 4 . . .
ǧ 4 4 . 4 . . _ 4 . . .
ṭ 4 4 . 4 . 4 4 _ . . .
l . . . . 4 . . . _ 4 .
L . . . . 4 . . . 4 _ .
ġ . . 3 5 1 1 . . . . _
''',
'''
_ a A * i y Y u w W
a _ 5 4 . . . . . .
A 7 _ 4 . . . . . .
* 4 4 _ . . . 3 . .
i . . . _ 3 3 . . .
y . . . 3 _ 5 . . .
Y . . . 6 5 _ . . .
u . 3 . . . . _ 3 3
w . . . . . . 5 _ 4
W . . . . . . 3 2 _
''',
'''
_ m n N w h ʼ ḥ h ḫ d ḏ ẓ l
m _ 3 2 . . . . . . . . . .
n 3 _ 4 . . . . . . . . . 3
N 2 4 _ . . . . . . . . . .
w . . . _ 3 3 . . . . . . .
h . . . 3 _ 3 3 . 4 . . . .
ʼ . . . 3 3 _ . 4 4 . . . .
ḥ . . . . 3 . _ 4 4 . . . .
h . . . . . 4 4 _ 4 . . . .
ḫ . . . . 4 4 4 4 _ . . . .
d . . . . . . . . . _ 4 . .
ḏ . . . . . . . . . . _ 4 .
ẓ . . . . . . . . . . 4 _ .
l . 3 . . . . . . . . . . _

'''
]


@dataclass
class MappedLetter:
    ltr : str
    lndx: int  # letter index
    repeat: int  # ltr repetition
    spos : int  # start position (ms)
    epos : int  # end position
    hipos : int = 0  # where the prediction is a maximum


def _test():
    br_nrm = BellRating(lslope=50, rslope=50, range=300)
    br_nrm.plot()
    br_len = BellRating(lslope=100, rslope=80, range=300)
    br_len.plot()
    br_pos = BellRating(lslope=70, rslope=120, range=300)
    br_pos.plot()
    for diff in range(-200, 200, 10):
        len_rtg = br_len.rate(diff)
        pos_rtg = br_pos.rate(diff)
        print(f"test rating - diff {diff:4d} : len {len_rtg:5.3f}, pos {pos_rtg:5.3f}")
    return

def main():
    #mm = Mimax(9999)
    #for v in [0.2, 0.1, 0.3, 0.2, 0.4, 0.6, 0.3]:
    #    mm.add(v)
    #print(mm.show())
    #return

    G.colors = colormix()
    G.runtoken = tbx.RunToken('map_multi.run_token')
    G.support_list = parse_support()  # collect all supporting neighbor relations into one list
    # print("support list: ", G.support_list)

    G.ratings = Counter()

    # prepare statistics file for collecting
    fn = f"ratings {dialog.name} ch_{dialog.chapno} bl_{dialog.blkno} {dialog.algo}.txt"
    fn = fn.replace('*', '#')
    G.statfile = cfg.work / dialog.recording / fn

    with open(G.statfile, mode='w') as fo:
        fo.write("ratings for some blocks\n\n")
        fo.write("xkey    text mapped  avg_rtg\n\n")

    pdb.db_connector(db_worker)

    print("saved ratings as", G.statfile)

def test():

    recd, chapno, blkno = 'hus9h', 2, 184

    vect_loader = att.VectLoader(recd, chapno, blkno)
    xdim = vect_loader.get_xdim()  # duration of block determins the x dimension of chart

    pyaa, orosa, freq = vect_loader.get_vectors()  # we use the librosa rms
    rosa = st.running_mean(orosa, 20, 3)  #
    pnp = get_peaks_and_pits(rosa, hyst=5)

    print(pnp)

def get_peaks_and_pits(seq, hyst=5):
    # seq is a list of amplitude values
    # hyst is a percentage of max peak as in indicator of a phase change
    pnp = []
    up = False
    for ndx, (a1, a2) in enumerate( zip(seq, seq[1:])):
        diff = abs(a1-a2)
        if up:
            if a1 > a2 and diff > 0.2:  # a1 is a peak
                pnp.append((ndx, 'k', round(a1, 2)))
                up = False
        else: # down
            if a1 < a2 and diff > 0.2:  # a1 is a pit
                pnp.append((ndx, 't', round(a1, 2)))
                up = True
    return pnp


def db_worker(dbman):
    stt = tbx.Statistics()

    dbref = dbman.connect(dialog.recording, 'proj')
    conn = dbref.conn

    ltrmap_json = cfg.work / 'mapped_letters.json'
    if os. path.exists(ltrmap_json):
        ltrmap = json.load(open(ltrmap_json, mode='r'))
    else:
        ltrmap = {}


    recd, chap_seq, blk_seq = dialog.recording, dialog.chapno, dialog.blkno
    model_name, iters = dialog.name, dialog.iters
    if iters == '*':
        specs = get_training_specs(recd, model_name)
        itno_list = range(specs.iter)
    else:
        tok = iters.split()  # iters is either a single number or a sequence of numbers (split by space)
        itno_list = [int(iters)] if len(tok) == 1 else [int(x) for x in tok]



    for chapno, blkno in tbx.chap_blk_iterator(recd, chap_seq, blk_seq):

        if G.runtoken.check_break():
            print("run token termination")
            break

        # if block is already done - execute anyway, if we dont plan to write the databse
        if check_block(conn, recd, chapno, blkno) and dialog.db_write:
            print(f"skipped prosessing for existing database data {chapno:03d}_{blkno:03d}")
            continue

        blk_results = []
        pred_loader = PredLoader(recd, chapno, blkno)
        vect_loader = att.VectLoader(recd, chapno, blkno)

        xdim = vect_loader.get_xdim()  # duration of block determins the x dimension of chart

        pyaa, orosa, freq = vect_loader.get_vectors()  # we use the librosa rms
        rosa = st.running_mean(orosa, 20, 3)  #
        G.pnplist = get_peaks_and_pits(rosa, hyst=5)

        text = prepare_text(recd, chapno, blkno, xdim)  # read and modify koran text

        print(f"\nblk: {chapno:03d}_{blkno:03d}  dura: {xdim-350}  text: [{text}] ({len(text)})\n")

        pred_list = prepare_prediction_data(itno_list, pred_loader, model_name)

        # all types of predictions are now lined up in one sequence
        try:

            mappings = map(pred_list, text, xdim)

        except Exception as excp:
            print("===== exception ====")
            traceback.print_exc()
            continue

        #print("mappings", mappings)

        # integrate mappings into json mappings
        # print("mappings 7")
        # {'ltr': 'y', 'lndx': 10, 'repeat': 1, 'search': 1569, 'ratg': 10.032699004466362, 'limit': 1811, 'lpos': 1631, 'rpos': 1808, 'itno': 12, 'mode': 'supp'}
        for ad in mappings:
            key = f"{chapno:3d} {blkno:3d} {ad.lndx:3d} {ad.ltr} {dialog.algo:2}"
            if ad.ratg == -1:
                data = (ad.repeat, 0, 0, ad.ratg)
            else:
                data = (ad.repeat, ad.lpos, ad.rpos, ad.ratg)
            ltrmap[key] = data


        # write statistics over mapping success
        rsum = 0   # sum of ratings
        count = 0  # number of rated letters
        miss = 0   # missing letters
        miss_ndx = []
        for ad in mappings:
            if ad.ratg == -1:  # indicates missing letter
                miss += ad.repeat
                miss_ndx.append(ad.lndx)
            else:
                rsum += ad.ratg * ad.repeat
                count += ad.repeat
        ravg = rsum / count

        save_ratings(chapno, blkno, len(text), count, ravg, miss, miss_ndx)


        block_info = AD(chapno=chapno, blkno=blkno, txtlen=len(text), mslen=xdim)
        blk_results.append((block_info, mappings))

        # # Write json files from the result data - not used now
        # fn = cfg.data / recd / 'mapping_results' /f"pred_results {chapno:03d}.{blkno:03d}.json"
        # with open(fn, mode='w') as fo:
        #    json.dump(blk_results, fo)

        # There should be a cleaning of the results, not every letter goes to the database

        if dialog.db_write:
            confirm_block(conn, recd, chapno, blkno)
            conn.commit()
            write_letterbase(conn, recd, blk_results)
        else:
            pass  # print(blk_results)

    # print_mapping_results(results)

    with codecs.open(ltrmap_json, 'w', encoding='utf-8') as fo:
        json.dump(ltrmap, fo, ensure_ascii=False, )
    print("saved lettermap json file")

    print(f"runtime: {stt.runtime()}")

def get_local_pnp(lpos, rpos):
    # just return the peaks and pits with in a certain time
    pnp = []
    for pos, typ, ampl in G.pnplist:
        if lpos < pos < rpos:
            pnp.append((pos, typ, ampl))
    return pnp

def map(pred_list, text, xdim):
    # pred list is a list of tuples: (itno, mode, avg_tab)
    # avg_tab is a dictionary with one vector per letter
    # letter_length is approx. ms per letter
    # xdim = total length in ms

    mapping_tab = []

    level = 0.07  # minimum probability, optimal is at 0.07

    curpos = 340  # Position is in ms - this is supposed to be the start of a letter
    curpos_f = 'in'  # initial
    old_repeat = 1
    ltrspec = AD(ratg = 0)
    G.linear_time = (xdim - 700) / len(text)

    for lndx, ltr, this_repeat in unique_letters(text):  # text is linear text with repetitions of certain vowels and mosaics

        # fist: calculate letter length (for the remaining letters)
        ltrs = len(text) - lndx   # remaining letters to map
        remaining = xdim - 350 - curpos  # remaining time (ms) in the block
        ltr_time = remaining / ltrs  # milliseconds per letter (unit)

        # next: determine the "current position" curpos
        if lndx == 0:

            ltrspec = scan_and_rate(pred_list, ltr, lndx, this_repeat, "in", curpos, curpos_f, curpos, ltr_time, level)

        else:
            # for the second and all later letters, there is the ltrspec of the previous letter

            if ltrspec.ratg == -1:
                advance = int(ltr_time * this_repeat)
                curpos += advance
                curpos_f = "er"
                backlimit = int(curpos - ltr_time * 0.1)

                ltrspec = scan_and_rate(pred_list, ltr, lndx, this_repeat, "er", curpos, curpos_f, backlimit, ltr_time, level)

            else:

                oltr, pl, pr, ratg = ltrspec.ltr, ltrspec.lpos, ltrspec.rpos, ltrspec.ratg  # these are from the previous letter

                # after the end of the letter is found, where to continue next search?
                # calculate an expected position based on
                # the current position, the remaining letters and the remaining time

                # pl is where this letter started, pr is where it ended
                pitpos = -1
                peakpos = -1
                pos4 = -1
                pnp = get_local_pnp(pl, pr)
                if pnp and dialog.usepnp:
                    print(f"found pnp at {pl}:{pr} ==> {pnp}")
                for loc, typ, ampl in pnp:
                    if typ == 't':
                        pitpos = loc
                    if typ == 'k':
                        peakpos = loc
                if iscons(oltr) and pitpos != -1:   # pref letter is consonat and has a pit
                    pos4 = pitpos
                if isvow(oltr) and peakpos != -1:   # pref letter is vowel and has a peak
                    pos4 = peakpos
                pos1 = int(curpos + ((old_repeat) * ltr_time))   # calculated next curpos
                #pos2 = int(curpos + )
                pos3 = pr
                if pos4 != -1 and dialog.usepnp:
                    mj_pos = pos4
                    mj_pos_f = 'pp'
                elif pos3 < pos1:
                    mj_pos = pos4
                    mj_pos_f = 'lp'
                else:
                    mj_pos = pos3
                    mj_pos_f = 'rp'

                mj_backlimit = mj_pos   # don't go back
                mj_pref_len = int(ltr_time * this_repeat)  # length with repetition, but a bit relaxed

                pos_au = int(curpos + old_repeat * ltr_time)
                pos_pr = int(pr - 20)  # the right boundary of the current letter

                if ratg > 10 and pos_au < pos_pr:
                    pos_au = pos_pr - 1

                if pos_pr < pos_au:
                    hb_pos = int(pos_pr)
                    hb_pos_f = "pr"
                    hb_backlimit = hb_pos - 10
                else:
                    hb_pos = int(pos_au)
                    hb_pos_f = "au"
                    hb_backlimit = pr - 30
                hb_pref_len = int(ltr_time * this_repeat - ltr_time * 0.1)  # length with repetition, but a bit relaxed

                if dialog.algo in ('*', 'mj'):
                    mj_ltrspec = scan_and_rate(pred_list, ltr, lndx, this_repeat, "mj", mj_pos, mj_pos_f, mj_backlimit, ltr_time, level, mj_pref_len)
                    mj_ratg = mj_ltrspec.ratg

                if dialog.algo in ('*', 'hb'):
                    hb_ltrspec = scan_and_rate(pred_list, ltr, lndx, this_repeat, "hb", hb_pos, hb_pos_f, hb_backlimit, ltr_time, level)
                    hb_ratg = hb_ltrspec.ratg

                if dialog.algo == '*':
                    if mj_ratg > hb_ratg:
                        curpos = mj_pos
                        ltrspec = mj_ltrspec
                    else:
                        curpos = hb_pos
                        ltrspec = hb_ltrspec
                    print(f"mj ratg: curpos {mj_pos} => {mj_ratg:5.2f},  hb ratg: curpos {hb_pos} => {hb_ratg:5.2f}  - new pos: {curpos}")

                elif dialog.algo == 'mj':
                    curpos = mj_pos
                    ltrspec = mj_ltrspec
                elif dialog.algo == 'hb':
                    curpos = hb_pos
                    ltrspec = hb_ltrspec
                else:
                    assert dialog.algo in ('mj', 'hb', '*')


        mapping_tab.append(ltrspec)

        if ltrspec.ratg > -1:
            last_pos = ltrspec.rpos

        old_repeat = this_repeat



    print(f"\ntext: [{text}]\n")
    print(f"       search ended at {curpos}, should end at {xdim-400}, unmapped audio: {xdim-400-last_pos}")

    return mapping_tab


def scan_and_rate(pred_list, ltr, lndx, repeat, algo,  curpos, curpos_f, backlimit, ltr_time, level, pref_len=-1):
    # these values are prepared for the rating
    if pref_len == -1:
        pref_len = int(ltr_time * repeat - ltr_time * 0.1)  # length with repetition, but a bit relaxed

    span = int(ltr_time * 1.5)  # time to find the start of the prediction
    # maybe the span should be different depending on letter and repeat

    targ_pos = curpos + 20 + int(pref_len / 2)  # rate the position by the middle

    # print(f"map ltr {ltr} *{repeat} ({lndx}) ltrs:{ltrs}  remain:{remaining} ltr_time={ltr_time:5.1f}  curpos:{curpos}  targ_pos:{targ_pos}")

    rate_it = PredictionRating(ltr, lndx, targ_pos, pref_len, pos_weight=4, len_weight=6, prob_weight=5)

    # the avg_tab is a dictionary of all letters with their probability vectors

    if dialog.debug:
        print(f"\nscan <{algo}> predictions: {ltr} ({lndx} *{repeat}) curpos:{curpos} "
              f"span:{span} (limit:{curpos + span}) unit:{ltr_time:6.1f} length:{pref_len} lvl:{level}")
    if lndx == 0:
        print("tested    rept ltr lndx  algo  curpos     lindev  intro    lpos  rpos  length   mpos   avgpr    pos/len/prob    rating    iteration ")
        #     #   32      1*  f (  0)   in    340 in     26>    366   533   ( 167)    449     0.72   3.98 5.68 3.62   17.27      26

    for itx, (itno, mode, avg_tab) in enumerate(pred_list):

        seq = avg_tab[ltr]  # select right probability vector for the letter
        if curpos >= len(seq): break

        # find a given letter in these predictions (of one iteration)
        # __________________________________________________________________________________________________
        for pl, pr, avg_prob in scan_predictions(itx, mode, ltr, lndx, seq, curpos, span=span, backlimit=backlimit, length=pref_len,
                                                 level=level):
            # print(f"it_result: itx {itx:2} pl {pl} pr {pr} avgprob {avg_prob:4.2f}")
            it_result = AD(pl=pl, pr=pr, avg_prob=avg_prob, itx=itx, itno=itno, mode=mode)
            it_result.pos = int((pr + pl) / 2)  # something like middle pos
            it_result.len = pr - pl
            it_result.curpos = curpos

            rate_it.add_object(it_result)
            # ltr_preds.append((itno, mode, it_result))

    best, num_of_ratgs = rate_it.get_best_object()

    linear_dev = -100 * (lndx * G.linear_time + 350 - curpos) / (len(seq)-700)
    data_1 = f"  {num_of_ratgs:3d}     {repeat:2d} *  {ltr} ({lndx:3d})  <{algo}>   {curpos:5d} {curpos_f}  {linear_dev:6.2f} "

    if num_of_ratgs == 0:
        print(data_1)
    else:
        intro = best.pl - curpos
        intro_s = f"{intro:5d}<" if intro < 0 else f"{intro:5d}>"

        data_2 = f"{intro_s}   {best.pl:5d} {best.pr:5d}  ({best.len:4d})  {best.pos:5d}   " \
                 f"{best.avg_prob:4.2f}  {best.pos_ratg:5.2f}/{best.len_ratg:5.2f}/{best.prob_ratg:5.2f}   " \
                 f"{best.ltr_ratg:5.2f}    {best.itno:2d} {best.mode} {best.itx:2} "
        print(data_1 + data_2)


    ltrspec = AD(ltr=ltr, lndx=lndx, repeat=repeat, search=curpos, ratg=-1)
    ltrspec.limit = int(curpos + span)  # define the end of the prediction search

    if not num_of_ratgs or best.pr == 0:  # if we find no prediction at all!
        return ltrspec   # add a dummy letter spec for an unmapped letter


    # {'pl', 'pr', 'hiprob', 'hipos', 'avg_prob', 'pos', 'len', 'itno', 'mode'}
    ltrspec.lpos = pl = best.pl
    ltrspec.rpos = pr = best.pr
    ltrspec.ratg = best.ltr_ratg
    ltrspec.itno = best.itno
    ltrspec.mode = best.mode

    # print(f"best: {pl, pr}, m:{int((pr+pl)/2)}  ratg:{best.ltr_ratg:5.2f}  it:{itno},{mode}")

    return ltrspec


def scan_predictions(itx, mode, ltr, lndx, seq, pos, span, backlimit, length, level):
    # seq = prediction vector for the given letter
    # pos = position in vector where to start the search
    # span = maximum range to search for the letter start
    # length = expected length of the letter
    # level = minimum probability of the predictions to be observed

    pred_tab = find_ranges(seq, pos, pos+span, backlimit, level)

    ''' The pred_tab may have 2 or 3 segments, where some probabilities were detected.
        Let's make this function a generator, so it can return more than one possible
        location of left and right letter boundaries
        
        Not too much thinking here. Yield the parts and combined parts as they are. Let the
        rating algorithm figure out, which the best boundaries are
    '''

    parts = []
    intro = 0
    for ndx, (pl, pr, avg) in enumerate(pred_tab):
        if ndx == 0:
            intro = pl - pos
        parts.append(f"{pl:5d} {pr:5d}  ({pr-pl:3d}) prb: {avg:4.2f}")
        if ndx+1 < len(pred_tab):
            pl = pred_tab[ndx+1][0]
            dist = pl - pr
            parts.append(f" <{dist:3}> ")

    if dialog.debug:
        print(f"pred_tab: x{itx:2} {mode}  {intro:5}> {'  '.join(parts)}")
    if len(pred_tab) > 0:
        yield pred_tab[0]
    if len(pred_tab) > 1:
        yield pred_tab[1]
        yield combine_pred_segments(pred_tab, 1)
    if len(pred_tab) > 2:
        yield combine_pred_segments(pred_tab, 2)


def combine_pred_segments(ptab, endx):
    # combine 2 or more prediction segments
    lpos = ptab[0][0]
    rpos = ptab[endx][1]
    sum_avg = 0
    for pl, pr, avg in ptab[:endx+1]:
        sum_avg += avg * (pr-pl)
    tavg = sum_avg / (rpos-lpos)
    return lpos, rpos, tavg


def find_ranges(seq, pos, scan_end, backlimit, level):
    # span is for the start of the search
    # length is for the end of search
    pred_tab = []

    # pos may already point to a place inside a prediction, in this case
    # let's search the left boundary
    while pos > backlimit and seq[pos] > level:
        pos -= 1

    inside = False
    pl = -1
    pr = -1
    sum_prob = 0
    p = pos - 1
    while True:
        p += 1

        if inside:
            if seq[p] > level:
                sum_prob += seq[p]
                continue

            inside = False
            pr = p
            avg_prob = sum_prob / (pr-pl)
            pred_tab.append((pl, pr, avg_prob))
            pl = pr = -1
            sum_prob = 0
            continue

        else:
            if p > scan_end or p >= len(seq):
                break

            if seq[p] > level:
                inside = True

                pl = p
                pr = -1

    if pl > -1:
        pred_tab.append((pl, pr, -1))

    # the pred_table usually contains only one range. if there is more than one range
    # pick the one, that is longer and has the higher avg probability
    # sometimes, there is a range without an end, this can be skipped
    # A letter may have a dip in the probability curve, so it essentially it should be
    # combined of two ranges
    return pred_tab


def save_ratings(chapno, blkno, tlen, mcount, ravg, miss, miss_ndx):
    # text_length, mapped count, rating average, miss count

    with open(G.statfile, mode='a') as fo:
        key = f"{chapno:03d}_{blkno:03d}"
        mtxt = f"miss: {miss:2d}  {miss_ndx}" if miss else ''
        fo.write(f"{key}  {tlen:3d}   {mcount:3d}   {ravg:6.3f}  {mtxt}\n")



class BellRating:
    def __init__(self, lslope=1, rslope=1, range=5):
        self.range = range
        self.steps = 200  # resolution on both sides of the curve
        self.stepsize = stepsize = range / self.steps
        lrange = np.arange(-range,0, stepsize)
        rrange = np.arange(0, range, stepsize)
        self.trange = np.arange(-range, range, stepsize)

        lcurve = norm.pdf(lrange, 0, lslope)
        rcurve = norm.pdf(rrange, 0, rslope) * (rslope/lslope)
        # the final curve gives 1.0 for the difference of zero, else some smaller value approaching zero
        self.tcurve = np.append(lcurve, rcurve) * (1 / lcurve[-1])


    def plot(self):
        # define plot
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(self.trange, self.tcurve, color="red")

        # choose plot style and display the bell curve
        plt.style.use('fivethirtyeight')
        plt.show()

    def rate(self, value):
        # print("ratg val:", value)
        ndx = int(value + self.steps)
        # print(f"steps: {self.steps}, ndx: {ndx}")
        if ndx >= len(self.tcurve) or ndx < 0:
            return 0
        ratg = self.tcurve[ndx]

        return ratg


class PredictionRating:
    def __init__(self, ltr, ltr_ndx, pref_pos, pref_len, pos_weight, len_weight, prob_weight):
        self.ltr = ltr
        self.ltr_ndx = ltr_ndx
        self.pref_pos = pref_pos  # this is, where the middle of the letter should be found
        self.pref_len = pref_len
        self.pos_weight = pos_weight
        self.len_weight = len_weight
        self.prob_weight = prob_weight
        self.obj_list = []
        # slope value is for flatness: small = steeper, large = flatter,
        # negative values go left, pos go right
        self.br_len = BellRating(lslope=100, rslope=80, range=300)
        self.br_pos = BellRating(lslope=70, rslope=120, range=300)

    def add_object(self, obj):
        # AD(pl, pr, hiprob, hipos, avg_prob, pos, len, itno, mode)
        self.obj_list.append(obj)
        return

    def rate_object(self, obj):
        # high rating is better
        pos_diff = obj.pos-self.pref_pos
        pos_ratg = self.br_pos.rate(pos_diff)

        len_diff = obj.len  - self.pref_len
        len_ratg = self.br_len.rate(len_diff)

        prob_ratg = obj.avg_prob
        pos_score = pos_ratg * self.pos_weight
        len_score = len_ratg * self.len_weight
        prob_score = prob_ratg * self.prob_weight
        # best rating is high, up to 23 (depends on the weights)

        #print(f"           rate obj  - pos diff: {pos_diff} {pos_ratg}  -  len diff:{len_diff} {len_ratg}  prob {prob_ratg}")
        return pos_score, len_score, prob_score

    def get_best_object(self):
        if len(self.obj_list) == 0:
            return None, 0
        valid_objs = 0
        for ndx, obj in enumerate(self.obj_list):
            if dialog.debug and ndx ==0:
                print(f"get best scan result for {self.ltr} ({self.ltr_ndx}) preferred pos {self.pref_pos}, len {self.pref_len}")

            if obj.pl:
                valid_objs += 1
                rtpos, rtlen, rtprob = self.rate_object(obj)
            else:
                rtpos, rtlen, rtprob = 0, 0, 0
            obj.pos_ratg = rtpos
            obj.len_ratg = rtlen
            obj.prob_ratg = rtprob
            ratg = (rtpos + rtlen + rtprob) * (1.3 if obj.mode == "orig" else 1.0)
            obj.ltr_ratg = ratg

        obj_rank = sorted(self.obj_list, reverse=True, key=lambda obj: obj.ltr_ratg)

        if dialog.debug:
            for obj in obj_rank[:10]:
                objstr = f"x:{obj.itx:2}  pos:{obj.pl:5d} : {obj.pr:5d}  m:{obj.pos:5d}  l:{obj.len:4d}  avgprob:{obj.avg_prob:4.2f}" \
                         f"  ratgs=> pos:{obj.pos_ratg:5.2f}  len:{obj.len_ratg:5.2f}  prob:{obj.prob_ratg:5.2f}   rtg:{obj.ltr_ratg:5.2f}"
                print("    rate: ", objstr)

        return obj_rank[0], valid_objs


def calculate_predictions_average(np_mat):
    # calculate a smoothed curve for the predictions
    # also turn the data into a letter-based dictionary
    avg_tab = {}  # smoothed single letter probabilities

    for ndx, ltr_seq in enumerate(np_mat):
        # put the averaged probability curves into a new matrix
        letter = G.lc.ltrs[ndx]
        avg = st.running_mean(ltr_seq, winsize=dialog.avgwin, iterations=3)
        avg_tab[letter] = avg

    return avg_tab


def prepare_text(recd, chapno, blkno, xdim):
    # the koran text does not match the reading length, some vowels and 'nasals' are
    # adjusted (added) to bring text and reading speed closer together
    text = tt.get_block_text(recd, chapno, blkno)  # , spec="ml")
    # print(f"db_text: [{text}] len: {len(text)}")

    text = text.strip('.').strip()  # remove '.', also remove space
    atext = adjust_timing(text)  # make text more linear in time

    # calulate relation between text and audio length
    # print(f"adjusted: [{atext}] len:{len(atext)}")
    return atext

def prepare_prediction_data(itno_list, pred_loader, model_name):

    pred_list = []
    # print(f"use model {model_name} with [{itno_list}] iterations")
    for itno in itno_list:

        ltr_rows = pred_loader.get_calculated_predictions(model_name, itno)
        # the predictions are for 5ms intervals - to avoid confusion, convert the data into 1ms
        ltr_rows = np.array([interpolate(row) for row in ltr_rows])
        for row in ltr_rows:  # each row has some noise: flatten 350 ms at both ends
            row[0:350] = 0
            row[-350:] = 0

        sup_rows = add_nbr_support(ltr_rows)
        # reversed_rows = np.flip(new_rows, axis=1)  # dont use the reverse approach

        # for each iteration of the model,
        #     try a mapping on the original predictions (ltr_rows)
        #     and on the supported predictions  (sup_rows)
        alt_mapping = (("orig", ltr_rows), ("supp", sup_rows))
        for mode, preds in alt_mapping:

            if not dialog.use_support and mode == "supp":
                # if not selected, skip supported mappings
                continue

            # print(f"chapno: {chapno:03d}, blkno: {blkno:03d}, iteration: {itno:2d},  mode: {mode}")

            avg_tab = calculate_predictions_average(preds)
            pred_list.append((itno, mode, avg_tab))
    return pred_list

"""
<<<<<<< HEAD
def save_ratings(chapno, blkno, tlen, mcount, ravg, miss, miss_ndx):
    # text_length, mapped count, rating average, miss count

    with open(G.statfile, mode='a') as fo:
        key = f"{chapno:03d}_{blkno:03d}"
        mtxt = f"miss: {miss:2d}  {miss_ndx}" if miss else ''
        fo.write(f"{key}  {tlen:3d}   {mcount:3d}   {ravg:6.3f}  {mtxt}\n")



def map(pred_list, text, xdim):
    # pred list is a list of tuples: (itno, mode, avg_tab)
    # avg_tab is a dictionary with one vector per letter
    # letter_length is approx. ms per letter
    # xdim = total length in ms

    mapping_tab = []
    ratg = 0  # total rating for this block
    # points for the rating observations
    ratg_notf  = 0

    level = 0.07  # minimum probability, optimal is at 0.07

    curpos = 430  # Position is in ms - this is supposed to be the start of a letter
    curpos_f = 'in'   # initial
    lndx = 0

    for ndx, ltr, repeat in unique_letters(text):  # text is linear text with repetitions of certain vowels and mosaics


        # G.debug = True if lndx in (12,86,) else False  # debug a certain letter

        if curpos >= xdim: # xdim is the total time of the block
            break

        # calculate expected letter length by dividing the remaining time by the remaining letters
        ltrs = len(text) - lndx   # remaining letters to map
        remaining = xdim - 350 - curpos  # remaining time (ms) in the block
        ltr_time = remaining / ltrs  # milliseconds per letter (unit)


        ltrspec = AD(ltr=ltr, lndx=lndx, repeat=repeat, search=curpos, lpos=0, rpos=0,
                     itno=-1, mode='x')

        # these values are prepared for the rating
        pref_len = int(repeat * ltr_time)  # length with repetition

        span = int(ltr_time * 1.2)   # time to find the start ogf the predictions
        ltrspec.limit = int(curpos + span)  # define the end of the prediction search
        targ_pos = curpos + 20 + int(pref_len / 2)  # rate the position by the middle

        # print(f"map ltr {ltr} *{repeat} ({lndx}) ltrs:{ltrs}  remain:{remaining} ltr_time={ltr_time:5.1f}  curpos:{curpos}  targ_pos:{targ_pos}")

        rate_it = PredictionRating(targ_pos, pref_len, pos_weight=5, len_weight=5, prob_weight=5)

        # the avg_tab is a dictionary of all letters with their probability vectors
        for itx, (itno, mode, avg_tab) in enumerate(pred_list):
            seq = avg_tab[ltr]  # select right probability vector for the letter

            # find a given letter in these predictions (of one iteration)
            it_result = scan_predictions(itx, ltr, lndx, seq, curpos, span=span, level=level)

            if it_result:  # may be None
                # got this: AD(ltr, ltr_ndx, pl, pr, hiprob, hipos, avg_prob)
                it_result.pos = int((it_result.pr + it_result.pl) / 2)  # something like middle pos
                it_result.len = it_result.pr - it_result.pl
                it_result.itno = itno
                it_result.mode = mode
                # it_result.curpos = curpos

                rate_it.add_object(it_result)
                #ltr_preds.append((itno, mode, it_result))

        best, num_of_ratgs = rate_it.get_best_object()

        advance = int(ltr_time * repeat)

        if num_of_ratgs:
            ratg_str = f"curp:{curpos:5d} {curpos_f} intro:{best.pl-curpos:3d} pl:{best.pl:5d}  len:{best.len:4d} ({repeat}) pr:{best.pr:5d} "\
                      f"mpos:{best.pos:5d} itno:{best.itno:2d} mode:{best.mode}  "\
                      f"ltr_ratg:{best.ltr_ratg:5.2f}  "\
                      f"pos/len/prob: {best.pos_ratg:5.2f}/{best.len_ratg:5.2f}/{best.prob_ratg:5.2f}  "\
                      f"avg_prob:{best.avg_prob:4.2f}  "  # \
                      # f"hi_prob:{best.hiprob:4.2f}  hipos:{best.hipos:5d}"
        else:
            ratg_str = f"curp:{curpos:5d} {curpos_f} --> {advance}"
        print(f"best: {num_of_ratgs:2d}*ratgs ==> "
              f"ltr: {best.ltr} ({best.lndx:3d})  {ratg_str} ")

        lndx += repeat  # letter index of the next letter


        if not num_of_ratgs or best.pr == 0:  # if we find no prediction at all!
            oldpos = curpos
            curpos += advance
            ratg += ratg_notf

            ltrspec.ltr_ratg = -1  # to indicate a missing letter
            mapping_tab.append(ltrspec)  # add a letter spec for an unmapped letter

            continue  # go to the next letter

        # {'pl', 'pr', 'hiprob', 'hipos', 'avg_prob', 'pos', 'len', 'itno', 'mode'}
        ltrspec.lpos = pl = best.pl
        ltrspec.rpos = pr = best.pr
        ltrspec.ltr_ratg = best.ltr_ratg
        ltrspec.itno = best.itno
        ltrspec.mode = best.mode

        # print(f"best: {pl, pr}, m:{int((pr+pl)/2)}  ratg:{best.ltr_ratg:5.2f}  it:{itno},{mode}")

        ratg += best.ltr_ratg

        mapping_tab.append(ltrspec)

        # after the end of the letter is found, where to continue next search?
        if dialog.algo == 'mj':
            # pl is where this letter started, pr is where it ended
            pos1 = pr
            pos3 = int(pl + (repeat * ltr_time))
            if pos1 > pos3:
                curpos = pos3
            else:
                curpos = pos1

        elif dialog.algo == 'hb':


            # calculate an expected position based on
            # the current position, the remaining letters and the remaining time

            nextpos_1 = curpos + repeat * ltr_time  - 0.5 * ltr_time
            # pl is where this letter started, pr is where it ended

            # nextpos_2 = int(pl + (repeat * ltr_time))  # referring to the left boundary is not useful
            #pos2 = (pos1 + letter_length)
            nextpos_3 = pr - 40  # the right boundary of the current letter
            #if pos1 - pos3 > letter_length:
            #    curpos = pos1
            #else:
            if nextpos_3 > nextpos_1:
                curpos = int(nextpos_3)
                curpos_f = "pr"
            else:
                curpos = int(nextpos_1)
                curpos_f = "au"
            # curpos = int(max(nextpos_1, nextpos_3))
        else:
            raise Exception(f"bad algorithm: {dialog.algo}")

    ratg_final = ratg / len(text)

    print(f"\ntext: [{text}]\n")
    print(f"       search ended at {curpos}, should end at {xdim-400}, unmapped audio: {xdim-400-curpos}")

    return mapping_tab, ratg_final

###itx, ltr, lndx, seq, curpos, pref_len+extra, level)
def scan_predictions(itx, ltr, lndx, seq, pos, span, level):
    # seq = prediction vector for the given letter
    # pos = position in vector where to start the search
    # minlen = minimum length of scanning - even when the predictions go (temporarily) down
    # level = minimum probability of the predictions to be observed
    highest = 0
    highpos = 0

    if G.debug and itx==0:
        print(f"scan predictions ({itx}): {ltr} {lndx} pos:{pos} lvl:{level}")

    limit = pos + span
    endpos = min(limit, len(seq)-1)  # don't run into the end of prediction vector

    # if predictions overlap, after one letter we may be in(!) the next letter already
    # so eventually go back

    # search the beginning of this letter (left boundary)
    lspan = 30 # go back but not further than span
    if seq[pos] >= level:
        pos = scan_left(ltr, seq, pos, lspan, level)


    pl = scan_right(ltr, seq, pos, endpos, level)

    if pl == 0:
        return AD(ltr=ltr, lndx=lndx, pl=0, pr=0, hiprob=0, hipos=0, avg_prob=0.0   )
        return None  # letter not found

    if G.debug:
        pass # print(f"   found {ltr} at {pl} prob: {seq[pl]}")
    # pl is the position of the left boundary
    # now go for the right boundary 'pr'
    endpos = min(pl + span, len(seq)-1)
    pr, highest, highpos, avgprob = scan_for_the_end(ltr, seq, pl, endpos, level)

    return AD(ltr=ltr, lndx=lndx, pl=pl, pr=pr, hiprob=highest, hipos=highpos, avg_prob=avgprob)


def scan_left(ltr, seq, pos, span, level):
    # call scan_left only if you are inside the letter
    lim = pos - span
    while seq[pos] >= level:  # go left for the left boundary
        pos -= 1
        if pos < lim:
            break
    return pos

def scan_right(ltr, seq, pos, endpos, level):
    # call scan_right only if you are before the letter start
    ipos = pos
    while pos < endpos:  # go right for the left boundary
        if seq[pos] >= level:
            break
        pos += 1
    else:
        return 0
    # print(f"      scan right '{ltr}' from {ipos}, found pl: {pos}, endpos={endpos}")
    return pos

def scan_for_the_end(ltr, seq, pos, endpos, level):
    # call scan_4_end to find the end of the letter
    # while minlen
    pl = pos # initial position = left boundary
    pr = 0
    highest = 0  # highest probability
    highpos = 0  # pos of the highest prob
    sumprob = 0  # add all probabilities to get the average

    scanpos = f"   scan for end {ltr} from {pos} to {endpos}"

    while True:
        # this code assumes that the probability of a letter may go down and come up to continue that letter
        prob = seq[pos]
        if prob >= level:
            sumprob += prob
            pr = pos # there is actually some probability here
            if prob > highest:
                highest = prob
                highpos = pos
        else:
            if G.debug:
                print(f"   {scanpos}   reached end of probability at {pos}")
            break
        pos += 1  # we continue through the sequence until endpos, even if the probabilities went down
        if pos > endpos:
            if G.debug:
                print(f"   {scanpos}   reached endpos at {pos}")
            break
    length = pr-pl
    avgprob = sumprob/length if length else 0
    return pr, highest, highpos, avgprob


class BellRating:
    def __init__(self, lslope=1, rslope=1, range=5):
        self.range = range
        self.steps = 100  # resolution on both sides of the curve
        self.stepsize = stepsize = range / self.steps
        lrange = np.arange(-range,0, stepsize)
        rrange = np.arange(0, range, stepsize)
        self.trange = np.arange(-range, range, stepsize)

        lcurve = norm.pdf(lrange, 0, lslope)
        rcurve = norm.pdf(rrange, 0, rslope) * (rslope/lslope)
        self.tcurve = np.append(lcurve, rcurve) * max(rslope, lslope) * 1.25


    def rate(self, value):
        # print("ratg val:", value)
        ndx = int(value + self.steps)
        # print(f"steps: {self.steps}, ndx: {ndx}")
        if ndx >= len(self.tcurve) or ndx < 0:
            return 0
        ratg = self.tcurve[ndx]

        return ratg


class PredictionRating:
    def __init__(self, pref_pos, pref_len, pos_weight, len_weight, prob_weight):
        self.pref_pos = pref_pos  # this is, where the middle of the letter should be found
        self.pref_len = pref_len
        self.pos_weight = pos_weight
        self.len_weight = len_weight
        self.prob_weight = prob_weight
        self.obj_list = []
        # slope value is for flatness: small = steeper, large = flatter,
        # negative values go left, pos go right
        self.br_len = BellRating(lslope=200, rslope=100, range=150)
        self.br_pos = BellRating(lslope=100, rslope=200, range=200)

    def add_object(self, obj):
        # AD(pl, pr, hiprob, hipos, avg_prob, pos, len, itno, mode)
        self.obj_list.append(obj)
        return

    def rate_object(self, obj):
        # high rating is better
        pos_diff = obj.pos-self.pref_pos
        pos_ratg = self.br_pos.rate(pos_diff)

        len_diff = obj.len  - self.pref_len
        len_ratg = self.br_len.rate(len_diff)

        prob_ratg = obj.avg_prob
        pos_score = pos_ratg * self.pos_weight
        len_score = len_ratg * self.len_weight
        prob_score = prob_ratg * self.prob_weight
        # best rating is high, up to 23 (depends on the weights)
        return pos_score, len_score, prob_score

    def get_best_object(self):
        if len(self.obj_list) == 0:
            return None, 0
        valid_objs = 0
        for ndx, obj in enumerate(self.obj_list):
            if G.debug and ndx ==0:
                print(f"get best scan result for {obj.ltr} ({obj.lndx}) preferred pos {self.pref_pos}, len {self.pref_len}")

            if obj.pl:
                valid_objs += 1
                rtpos, rtlen, rtprob = self.rate_object(obj)
            else:
                rtpos, rtlen, rtprob = 0, 0, 0
            obj.pos_ratg = rtpos
            obj.len_ratg = rtlen
            obj.prob_ratg = rtprob
            obj.ltr_ratg = rtpos + rtlen + rtprob
            if G.debug:
                objstr = f"pos:{obj.pl:5d}-{obj.pr:5d}  mpos:{obj.pos:5d}  len:{obj.len:4d}  avgprob:{obj.avg_prob:4.2f}" \
                         f"  ratgs=> pos:{obj.pos_ratg:5.2f} len:{obj.len_ratg:5.2f} prob:{obj.prob_ratg:5.2f} ltr:{obj.ltr_ratg:5.2f}"
                print("    rate: ", objstr)

        obj_rank = sorted(self.obj_list, reverse=True, key=lambda obj: obj.ltr_ratg)

        return obj_rank[0], valid_objs


def calculate_predictions_average(np_mat):
    # calculate a smoothed curve for the predictions
    # also turn the data into a letter-based dictionary
    avg_tab = {}  # smoothed single letter probabilities

    for ndx, ltr_seq in enumerate(np_mat):
        # put the averaged probability curves into a new matrix
        letter = G.lc.ltrs[ndx]
        avg = st.running_mean(ltr_seq, winsize=dialog.avgwin, iterations=3)
        avg_tab[letter] = avg

    return avg_tab


def get_weights(line, colx):
    # get the weights from the line,
    benf = line[0] # the beneficiary letter is in column 0
    for ndx, ltr in enumerate(line):
        if not ltr in "._ " and ndx > 1:
            col = int(ndx/2)
            nbr = colx[col]  # the neighbor who gives support
            pcnt = support_weights[int(ltr)] / 100
            yield nbr, benf, pcnt

def get_columns(line):
    # dictionary of letters for each column
    cols = {}
    for ndx, ltr in enumerate(line):
        if ltr not in "_ ":
            col = int(ndx/2) # this column index refers to the line index
            cols[col] = ltr
    return cols

def parse_lines(mat):
    # basic check of lines, yield lines one by one
    lines = [l.strip() for l in mat.splitlines() if l.strip()]
    if lines[0] == '#':
        return
    len0 = len(lines[0])
    cols = int(len0/2) + 1
    assert len(lines) == cols, f"len of line 0: {len0} does not match number of lines: {len(lines)}"
    for l in lines:
        assert len(l) == len0, f"line:{l} does not match the length {len0} of the first line"
        yield l

def prepare_text(recd, chapno, blkno, xdim):
    # the koran text does not match the reading length, some vowels and 'nasals' are
    # adjusted (added) to bring text and reading speed closer together
    text = tt.get_block_text(recd, chapno, blkno)  # , spec="ml")
    # print(f"db_text: [{text}] len: {len(text)}")

    text = text.strip('.').strip()  # remove '.', also remove space
    atext = adjust_timing(text)  # make text more linear in time

    # calulate relation between text and audio length
    # print(f"adjusted: [{atext}] len:{len(atext)}")
    return atext


def add_nbr_support(ltr_rows):
    new_rows = np.copy(ltr_rows)
    # print("new_rows:", new_rows.shape)
    for nbr, benf, pcnt in G.support_list:
        nbrx = G.lc.categ[nbr]
        benx = G.lc.categ[benf]
        # print(f"nbr_supp: {nbr}:{nbrx}, {benf}:{benx}")
        new_rows[benx] += new_rows[nbrx]*pcnt
    return new_rows

=======
>>>>>>> 7aab183cafd88ead1de17afe44ce7702345de044
"""

def write_results_file(chapno):
    # write rating results as a matrix
    blkdir = {}
    blks = set()
    iters = set()
    for blk, itno, ratg, mapping in G.results:
        blks.add(blk)
        iters.add(itno)
        if not blk in blkdir:
            blkdir[blk] = {}
        blkdir[blk][itno] = ratg
    itersum = {k:0 for k in iters}
    blksum = 0
    lines = []
    itstr = "  ".join([f"{itno:6d}" for itno in sorted(iters)])
    lines.append(f'blkno    avg  {itstr}')

    for blk in sorted(blks):

        for itno in sorted(iters):
            ratg = blkdir[blk][itno]
            blksum += ratg
            itersum[itno] += ratg
        iterline = ' '.join([f"{blkdir[blk][itno]:5.2f}  " for it in sorted(iters)])
        lines.append(f"  {blk:03d}  {blksum/len(iters):5.2f}    {iterline}")
    iterline = ' '.join([f" {itersum[itno]/len(blks):5.2f} " for itno in iters])
    lines.append(f"  avg:        {iterline}")

    result_fn = cfg.work / dialog.recording / f"results_{chapno}_{dialog.blkno}.txt"
    with open(result_fn, mode='w') as fo:
        for line in lines:
            fo.write(f"{line}\n")


def write_letterbase(conn, recd, results):
    for blkinfo, mappings in results:
        print(f"write db {blkinfo.chapno}-{blkinfo.blkno} mappings: {len(mappings)}")
        cbkey = f"{blkinfo.chapno:03d}_{blkinfo.blkno:03d}"
        sql = 'DELETE FROM lettermap WHERE recd == ? and cbkey == ?'
        conn.execute(sql, (recd, cbkey))

        # at the very ends of the block set the gaps to zero
        mappings[0].lgap = 0
        mappings[-1].rgap = 0
        for l1, l2 in zip(mappings, mappings[1:]):
            l1.rgap = l2.lgap = l2.lpos - l1.rpos

        sql = """INSERT INTO lettermap (ltr, recd, cbkey, lndx, rept, ratg, 
                 lpos, rpos, tlen, lgap, rgap)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        print("write letter mappings", mappings)
        for ltr in mappings:
            tlen = ltr.rpos - ltr.lpos
            ratg = round(ltr.ratg, 2)

            conn.execute(sql, (ltr.ltr, recd, cbkey, ltr.lndx, ltr.repeat, ratg,
                               ltr.lpos, ltr.rpos, tlen, ltr.lgap, ltr.rgap))
    return

'''
    ltr    text  nn  # individual letter
    recd   text  nn  # recording
    cbkey  text  nn
    lndx   int   nn  # letter index of the first letter 
    rept   int   nn  # repetition: add this to the current lndx to get the next lndx
    ratg   float     # some value, which represents the 'quality' of the letter, higher is better
    lpos   int   nn  # left position (ms)
    rpos   int   nn  # right position (ms)
    lgap   int   nn  # gap to the previous letter (negative if overlap)
    rgap   int   nn  # gap to the next letter (negative if overlap)
    lampl  int       # amplitude (loudness) at the left boundary
    rampl  int       # amplitude (loudness) at the right boundary
    lfreq  int       # freq at the left boundary
    rfreq  int       # freq at the right boundary
    melody text      # text represents a number of intermediate frequencies

'''

def check_block(conn, recd, chapno, blkno):
    # return true if block is already in this database table
    sql = "SELECT blkno from mapping_done WHERE recd == ? and chapno == ? and blkno == ?"
    csr = conn.execute(sql, (recd, chapno, blkno))
    rows = csr.fetchall()
    return len(rows) > 0

def confirm_block(conn, recd, chapno, blkno):
    # insert this block to confirm, it is already done
    sql = "INSERT INTO mapping_done (recd, chapno,blkno) VALUES (?, ?, ?)"
    conn.execute(sql, (recd, chapno, blkno))

#--------------------------------------------
# code to calculate the letter support


def add_nbr_support(ltr_rows):
    new_rows = np.copy(ltr_rows)
    # print("new_rows:", new_rows.shape)
    for nbr, benf, pcnt in G.support_list:
        nbrx = G.lc.categ[nbr]
        benx = G.lc.categ[benf]
        # print(f"nbr_supp: {nbr}:{nbrx}, {benf}:{benx}")
        new_rows[benx] += new_rows[nbrx]*pcnt
    return new_rows


def parse_support():
    # build a list of support weights
    support_list = []
    for matx, mat in enumerate(friends):
        for lx, line in enumerate(parse_lines(mat)):
            if lx == 0:
                colx = get_columns(line)
            else:
                for weight in get_weights(line, colx):
                    nbr, benf, pcnt = weight
                    support_list.append((nbr, benf, pcnt))
    return support_list


def parse_lines(mat):
    # basic check of lines, yield lines one by one
    lines = [l.strip() for l in mat.splitlines() if l.strip()]
    if lines[0] == '#':
        return
    len0 = len(lines[0])
    cols = int(len0/2) + 1
    assert len(lines) == cols, f"len of line 0: {len0} does not match number of lines: {len(lines)}"
    for l in lines:
        assert len(l) == len0, f"line:{l} does not match the length {len0} of the first line"
        yield l

def get_columns(line):
    # dictionary of letters for each column
    cols = {}
    for ndx, ltr in enumerate(line):
        if ltr not in "_ ":
            col = int(ndx/2) # this column index refers to the line index
            cols[col] = ltr
    return cols


def get_weights(line, colx):
    # get the weights from the line,
    benf = line[0] # the beneficiary letter is in column 0
    for ndx, ltr in enumerate(line):
        if not ltr in "._ " and ndx > 1:
            col = int(ndx/2)
            nbr = colx[col]  # the neighbor who gives support
            pcnt = support_weights[int(ltr)] / 100
            yield nbr, benf, pcnt



def adjust_timing(text):
    # make text more linear (in time) by inserting extra letters
    newtxt = []
    for l1, l2 in zip(text, text[1:]):
        newtxt.append(l1)
        if isvow(l1) and isvow(l2):
            newtxt.append(l1)
        if iscons(l1) and iscons(l2):
            newtxt.append(l2)
    newtxt.append(l2)
    newtxt = ''.join(newtxt)
    newtxt = newtxt.replace('..', '.')
    return newtxt

def isvow(l):
    return l in vowels
def iscons(l):
    return l not in vowels

def unique_letters(text):
    # A generator - yield repeated letters only once, but with repeat counter
    skip = 0
    for ndx, ltr in enumerate(text):
        if skip:
            # skip over previously processed letters
            skip -= 1
            continue

        # double letters are detected only once, but are counted
        dndx = ndx
        repeat = 1
        while True:
            dndx += 1
            if dndx < len(text) and text[dndx] == ltr:
                repeat += 1
                skip += 1
                continue
            else:
                break
        yield ndx, ltr, repeat


def interpolate(vect_5ms, yfact=1, move=0):
    # our data comes at 5ms intervals. For the plotting we want the data in 1 ms interval
    # let numpy do the interpolation
    # eventually apply some data shifting
    # move is the where the vector data is shifted
    #   +10 means insert 10 ms at the beginning
    #   -10 means remove 10 ms from the beginning
    l = len(vect_5ms)
    x      = np.linspace(0, l, num= l * 5)
    points = np.linspace(0, l, num= l)
    vect_1ms = np.interp(x, points, vect_5ms)
    vect_1ms *= yfact
    if move < 0:
        final = vect_1ms[-move:]
    elif move > 0:
        vlen = len(vect_1ms)
        final = np.ndarray((vlen + move,), dtype="float")
        final[move:vlen + move] = vect_1ms
    else:
        final = vect_1ms

    return final


def colormix():
    cormix = []
    val = [200, 150, 100, 50]
    for r,g,b in product(val, val, val):
        if 350 <= (r+g+b) <= 500:
            cormix.append((r/255,g/255,b/255))
    #print("colormix, found:", len(cormix))
    cortab = {l : c for l, c in zip(G.lc.ltrs, cormix)}
    return cortab

main()
#test()
