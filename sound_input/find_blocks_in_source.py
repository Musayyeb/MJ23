# python3
"""
    read the source audio files, which are a wav files per chapter
    to get the loudness, there is some audio processing first
    The blocks are identified by parameters like min_duration or
    averaging window size
    The output of this module goes to a text file, (overwrite)
"""


from config import get_config
cfg = get_config()
import splib.sound_tools as st
from splib.toolbox import get_block_counters
from splib.cute_dialog import start_dialog
import splib.toolbox as tbx
from splib.sound_input_lib import get_amplitude_average
from splib.sound_input_lib import FixBlocks
from matplotlib import pyplot as plt
from itertools import zip_longest
import numpy as np
import time

average_loudness = 2500  # used to calculate a factor for the max_ampl

class dialog:
    recording = "hus1h"
    chapter = 0
    winsize   = 300  # ms window for averaging, 300 ms is reasonable
    iter_avg = 2     # repeat averaging function
    max_ampl = 200   # amplitude limit for gap noise
    min_gaplen = 280 # anything shorter is probably a plosive
    min_blklen = 400 # anything shorter is probably a noise


    layout = """
    title   Convert video file into wav format
    text    recording  recording id, example: 'hus1h'
    text    chapter    koran chapter number or "*" (all)
    label   * for the calculating the average *
    label   * ______  hus0h  hus1h   hus2h   hus9h
    label   * mingap    200,,, 150,,,  210,,,  1200
    int     winsize    window size (ms) for the average
    int     iter_avg   number of iterations for the avg function
    label   * for the detection of block boundaries *
    int     min_blklen   minimum length of a block
    int     min_gaplen   minimum duration of a gap
    int     max_ampl     maximum amplitude in a gap
    """

def main():
    if not start_dialog(__file__, dialog):
        return

    stt = tbx.Statistics()

    # the dialog window may specify a specific block (number)
    # or "*" for all available blocks
    if dialog.chapter == "*":
        sel_chap = None  # which means all
    else:
        sel_chap = int(dialog.chapter)

    recd = dialog.recording
    blk_counters = get_block_counters(recd)

    fb = FixBlocks(cfg.work)
    if not fb.load(recd, reset=True):
        raise Exception("Could not load/create json data")

    extra = miss = extsum = missum = 0

    for fn, full in tbx.locate_data_files("source", recd=recd, chap=sel_chap):
        print(f"\nLoad: {full}")

        chapno = int("".join([x for x in fn if x in "0123456789"]))
        text_blocks = blk_counters[chapno]
        fr, wav = st.read_wav_file(full)

        # process one chapter


        # get an average over a small window size, to eliminate small peaks
        avg, total_loudness = get_amplitude_average(wav, 50, 2)
        max_ampl = int(dialog.max_ampl * (total_loudness/average_loudness))
        print(f"length of wav {len(wav)}, avg {len(avg)}")
        # print(f'block_loudness: {total_loudness}, modified max_ampl: {max_ampl}')


        # elim_count = eliminate_noise(avg)
        # print('    noise elimination count', elim_count)
        # a smoother average from the dialog parameters
        avg = st.running_mean(avg, dialog.winsize, dialog.iter_avg)

        veclen = len(avg)
        # then find the blocks

        blocks, raw_ct = find_blocks(avg, max_ampl)  # raw amplitude scan
        #print("blocks",blocks)
        #print("count", raw_ct)

        blockl = blocks_w_lengths(blocks)  # add length ot each block / gap

        blocks_2, noise_ct = eliminate_noise(blockl, dialog.min_blklen)

        blocks_3, gap_ct = eliminate_smallgaps(blocks_2, dialog.min_gaplen)

        blocks_4, no1st = blocks_3, gap_ct

        if dialog.recording == "hus9h":
            if not chapno in (9,95):
                blocks_4, no1st = eliminate_first(blocks_3)
        elif dialog.recording == "hus1h":
            if chapno not in (9, 60):
                blocks_4, no1st = eliminate_first(blocks_3)
            if chapno in (1,10,13,15,23,26):
                blocks_4, no1st = eliminate_first(blocks_4)

        final = no1st

        blocks = [blk[1] for blk in blocks_4[1:]]  # only save the position, ignore 1st gap
        #print("json blocks",blocks)
        fb.add_blocks(chapno, blocks)

        diff = final - text_blocks

        print(f'text={text_blocks:3d}, raw={raw_ct:3d}, nonoise={noise_ct:3d}, smallgap={gap_ct},'
              f' no1st={no1st:3d}    diff={diff}')


        if dialog.chapter != '*':
            fig = plt.figure(figsize=(18,6))
            fig.tight_layout()
            plt.plot(np.array([0]*veclen), color='silver')
            plt.plot(avg, color='black')
            plt.plot(plotcurve(blockl, veclen, 6,4), color='green') # raw amplitude
            plt.plot(plotcurve(blocks_2, veclen, 5,3), color='blue')  # eliminate noise
            plt.plot(plotcurve(blocks_4, veclen, 4,1), color='red')   # small gaps
            plt.show()


        if diff > 0:
            extra += 1
            extsum += diff
        if diff < 0:
            miss += 1
            missum -= diff


    fb.save()

    msg = f'''
    Find blocks in source  {tbx.now()}
        recording: {dialog.recording}
        avg_winsize: {dialog.winsize}
        min_gaplen: {dialog.min_gaplen}
        min_blklen: {dialog.min_blklen}
        max_ampl: {dialog.max_ampl}
    
    chapter: {dialog.chapter}
    {extra} chapters with extra blocks found: {extsum}
    {miss} chapters with missing blocks: {missum}
    '''
    print(msg)


    protofile = cfg.work / recd / "find_blocks_results.txt"
    with open(protofile, mode='a') as fo:
        fo.write(msg)
    print(f"***** see protocol: {protofile} *****")

    print(stt.runtime())

def plotcurve(blocks, vlen, lo, hi):
    #print('plot curve, len=', vlen)
    lo, hi = lo*-1000, hi*-1000
    vect = np.array([lo]* vlen)
    for typ, pos, blen in blocks:
        if typ == 'b':
            vect[pos:pos+blen] = hi
    return vect

def blocks_w_lengths(blocks):
    # block table comes without length, add length to all blocks and gaps
    nblocks = []
    prevpos = -1
    prevtyp = 'x'
    for typ, pos in blocks:
        if prevtyp != 'x':
            nblocks.append((prevtyp, prevpos, pos-prevpos))
        prevtyp, prevpos = typ, pos
    nblocks.append(('x', pos, 0))
    return nblocks

def eliminate_noise(blocks, minblk):
    # eliminate all too small blocks, which are mostly click noises
    dummy = ('#', 0, 0)
    for ndx, (typ, pos, siz) in enumerate(blocks):
        if typ == 'g':
            prev_gap = pos
        if typ == 'b' and siz < minblk:
            ntyp, npos, nsiz = blocks[ndx+1]
            blocks[ndx-1] = blocks[ndx] = dummy
            blocks[ndx+1] = (ntyp, prev_gap, npos - prev_gap + nsiz)
    nblocks = [x for x in blocks if x[0] != '#'] # remove all dummy values
    blkct = len(blocks)//2 - 1  # its gbg....bgx  always one gap  more
    return nblocks, blkct

def eliminate_smallgaps(blocks, mingap):
    # eliminate all too small blocks
    dummy = ('#', 0, 0)
    prev_blk = 0
    for ndx, (typ, pos, siz) in enumerate(blocks):
        if ndx == 0:
            continue
        if typ == 'b':
            prev_blk = pos
        if typ == 'g' and siz < mingap:
            nextyp = blocks[ndx+1][0]
            if nextyp != 'x':  # the last gap may be short
                ntyp, npos, nsiz = blocks[ndx+1]
                blocks[ndx-1] = blocks[ndx] = dummy
                blocks[ndx+1] = (ntyp, prev_blk, npos - prev_blk + nsiz)
    nblocks = [x for x in blocks if x[0] != '#'] # remove all dummy values
    blkct = len(blocks)//2 - 1  # its gbg....bgx  always one gap  more
    return nblocks, blkct


def eliminate_first(blocks):
    # block list starts with gap, blk, gap, blk, gap, ....
    # the first block is in the audio files, but not in the text
    # remove first block and put a long gap instead

    typ, pos, blen = blocks[0]
    # print(f"eliminate first, old: {typ, pos, blen}")

    nblocks = blocks[2:]
    typ, pos, blen = nblocks[0]
    nblocks[0] = typ, 0, pos+blen
    blkct = len(nblocks)//2 - 1  # its gbg....bgx - don't count the X

    typ, pos, blen = nblocks[0]
    # print(f"eliminate first, new: {typ, pos, blen}")

    return nblocks, blkct



def find_blocks(loud, amp_limit):
    # get the most simple block boundaries list
    # no length -lengths can be derived from the list

    blocks = []
    hyst = 0.9
    hyslo, hyshi = int(amp_limit*hyst), int(amp_limit/hyst)
    blocks.append(('g', 0))
    inblk = False
    for ndx, a in enumerate(loud):
        if inblk:
            if a < hyslo: # loud enough
                blocks.append(('g', ndx))
                inblk = False
        else:  # in gap
            if a > hyshi: # low enough
                blocks.append(('b', ndx))
                inblk = True
    blocks.append(('x', ndx))
    blkct = len(blocks)//2 - 1  # its gbg....bgx  always one gap  more
    return blocks, blkct

def __find_blocks(loud, max_ampl, min_gaplen):
    blocks = []

    blk_sizes = []
    gap_sizes = []
    g = True
    gap_start = 0
    blk_start = 0
    first = True
    for ndx, a in enumerate(list(loud)):
        if g:  # inside a gap
            if a < max_ampl:
                continue
            else: # transit - end of gap = begin of block
                g = False
                dura = ndx - gap_start
                if dura < min_gaplen and not first:  # ignore min_gaplen for the first gap
                    continue  # gap is too short
                blk_time = gap_start - blk_start
                if not first:
                    if blk_time > dialog.min_blklen:
                        blocks.append(('b', blk_start, blk_time ))
                        blk_sizes.append(blk_time)
                    else:
                        gap_start = prev_gap
                first = False
                blocks.append(('g', gap_start, dura))
                gap_sizes.append(dura)
                blk_start = gap_start + dura
                gap_start = ndx
        else: # inside a block
            if a > max_ampl:
                continue
            else:  # transit - begin of gap = end of block
                dura = ndx - gap_start
                #gaps.append(('b', pos, dura))
                g = True
                prev_gap = gap_start
                gap_start = ndx

    blk_time = gap_start - blk_start
    blocks.append(('b', blk_start, blk_time))
    blk_sizes.append(blk_time)
    dura = ndx - gap_start
    blocks.append(('g', gap_start, dura))
    gap_sizes.append(dura)

    # print("   shortest blocks (s):", [round(x/1000,3) for x in sorted(blk_sizes)[:10]])
    # print("   longest  blocks (s):", [round(x/1000,3) for x in sorted(blk_sizes)[-10:]])
    # print("   shortest gaps (ms):", [int(x) for x in sorted(gap_sizes)[:10]])

    return blocks


def __eliminate_noise(avg):
    # put silence to the loundess vector, where a noise (small peak) is identified
    g2b = 275  # use a hysteresis: a higher value for the gap to block transition
    b2g = 100  # a lover value for the block to gap transition

    gap = True
    gap_start = 0
    blk_start = 0
    max_a = 0

    seq = []

    # collect blocks and gaps and for the blocks also the maximum amplitude
    def gap_to_block(pos, gap_start):
        seq.append(('g', gap_start, max_a))  # save the prev gap
        return False, pos  # gap_flag, block_start

    def block_to_gap(pos, blk_start):
        seq.append(('b', blk_start, max_a))
        return True, pos  # gap_flag, gap_start, which may be unchanged

    for ndx, a in enumerate(list(avg)):
        if gap:
            if a > g2b:
                gap, blk_start = gap_to_block(ndx, gap_start)
        else:  # block
            max_a = max(max_a, a)
            if a < b2g:
                gap, gap_start = block_to_gap(ndx, blk_start)
                max_a = 0
    gap_to_block(ndx, gap_start)  # to identify the last gap

    # check size, amplitude and size of left and right gaps for each block

    elim_count = 0

    for g1, b1, g2, b2 in zip_longest(seq, seq[1:], seq[2:], seq[3:]):
        t, pos1, _ = g1
        if t != 'g': continue
        if b1 is None: break
        _, posb, max_a = b1
        t, pos2, _ = g2
        assert t == 'g'
        if b2 is None:
            end = pos2 + 500
        else:
            _, end, _ = b2

        l1 = posb - pos1  # length gap1
        bl = pos2 - posb  # length block
        l2 = end - pos2  # length gap2

        el = False
        if bl < 600:
            if l1 > 1000 and l2 > 1000:
                el = True
            elif max_a < 1600:
                el = True

        # if a 'noise' is identified, put zeroes to the loudness vector
        if el:
            elim_count += 1
            sec, ms = divmod(pos2, 1000)
            t = time.strftime('%H:%M:%S', time.gmtime(sec))
            # print(f'{pos2:9d}  {t}.{ms:03d}  lg{l1:5d}   bl={bl:5d} ({max_a:5d})  rg{l2:5d}')
            for p in range(posb, posb + bl):
                avg[p] = 0

    return elim_count


main()