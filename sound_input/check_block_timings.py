#python3
"""
    Check Block timings

    Make sure the block boundaries are correct
    Make a comparison between the audio length of a block
    and the text length.
"""
import splib.toolbox
from config import get_config
cfg = get_config()
import splib.text_tools as tt
from splib.cute_dialog import start_dialog
import splib.toolbox as tbx
from splib.sound_input_lib import FixBlocks, get_amplitude_average
import time

class G:
    block_ct = {}  # number of blocks per chapter
    recd = ''
    chap = ''
    fixblocks = None

    block_dict = {}  # key = blkno, val = start, duration
    block_ends = []  # ms positions of the end of the blocks
    infinity = 9e12  # millisecond position long after the end of the audio

class dialog:
    recording = "hus1h"
    chapter = 0


    layout = """
    title   Check Block Timings
    text    recording  recording id, example: 'hus1h'
    text    chapter    koran chapter number or "*" (all)
    """

def main():
    if not start_dialog(__file__, dialog):
        return
    stt = tbx.Statistics()

    # the dialog window may specify a specific block (number)
    # or "*" for all available blocks
    if dialog.chapter == "*":
        chap_seq = range(1, 115)
    else:
        chap_seq = [int(dialog.chapter)]


    G.recd = dialog.recording

    G.fixblocks = FixBlocks(cfg.work)
    G.fixblocks.load(G.recd)

    G.block_ct = splib.toolbox.get_block_counters(G.recd)

    for chap in chap_seq:
        # process one chapter at a time
        print(f"\n      process chapter {chap}\n")


        read_block_boundaries(chap)

        avg_tempo = check_avg_time_per_unit(G.recd, chap)

        show(G.recd, chap, G.block_dict, avg_tempo)

    print(stt.runtime())

def read_block_boundaries(chap):
    # the block boundaries are taken from a json file
    # this json file is managed by the FixBlocks Object

    block_dict = {}
    block_ends = []
    blocktab = G.fixblocks.get_fixed_blocks(chap)

    # blocks_wo = [(t, p) for t,p,l in blocktab]  # blocktab withouth lengths
    # G.fixblocks.add_blocks(chap, blocks_wo)  # we NEVER update the blocks in Fixblocks

    blkx = 0
    for typ, pos, len in blocktab:
        if typ == 'b':
            blkx += 1
            block_dict[blkx] = pos, len
        if typ == 'g':
            if blkx == 0:
                block_ends.append((len-50, blkx))
            else:
                block_ends.append((pos, blkx))
        if typ == "x": # for the "dummy" block after the last gap
            block_ends.append((pos, 999))

    G.block_dict = block_dict
    G.block_ends = block_ends
    #print(f"chap {chap} block dict", block_dict)
    #print(f"block endings:", block_ends)
    return


def get_fixes(chap):
    # a generator, which return the next place, where a fix has to happen
    fixes = G.fixblocks.get_fixes(chap)
    for pos, item in sorted(fixes):
        yield pos, item
    yield G.infinity, 'dummy'


def apply_fixes(chap, blocktab):
    # blocktab comes with (typ, pos, len)
    # return a new block tab with changed block/gap boundaries

    fixtab = get_fixes(chap)

    while True:

        newblocks = []

        fixpos, fixaction = next(fixtab)
        print(f"Fix Action to apply: {fixpos}  {fixaction}")
        #print('blocktab:', blocktab)

        if fixpos == G.infinity:
            newblocks = blocktab
            break


        for ndx, ((t1, p1, l1), (t2, p2, l2)) in enumerate(zip(blocktab, blocktab[1:])):

            # t, p, l = type (b/g), pos, length
            if p2 < fixpos:
                newblocks.append((t1, p1, l1))
                continue

            # we detected a fix for item 1

            if fixaction in ('nogap', 'noblk'):
                # actally the same action, independent of the type
                t0, p0, l0 = newblocks[-1]
                newblocks[-1] = ((t0, p0, l0+l1+l2))

                newblocks.extend(blocktab[ndx+2:])  # the current t1 is skipped
                break

            elif t1 == 'b':
                if fixaction == 'isgap':
                    # we could go into the audio, and find the silent part
                    # for now I (Hans) am too lazy - just insert a short gap
                    hgap = 20  # half gap
                    lnew = fixpos - p1 - hgap
                    newblocks.append((t1, p1, lnew))
                    newblocks.append(('g', fixpos-hgap, hgap+hgap))
                    lnew = p2 - fixpos+hgap
                    newblocks.append(('b', fixpos+hgap, lnew))

                    newblocks.extend(blocktab[ndx+1:])
                    break

        else:
            # some fix action is not processed (???)
            newblocks.append((t2, p2, l2))

        blocktab = newblocks
        #print('blocktab:', blocktab)

    return newblocks


def blocks_w_lengths(blocks):
    print("blocks w length in", blocks[:6])
    # block table comes only with positions, add type and length
    nblocks = []
    prevpos = 0
    prevtyp = 'b'
    for pos in blocks:
        typ = 'g' if prevtyp == 'b' else 'b'  # start with a gap
        nblocks.append((typ, prevpos, pos - prevpos))
        prevtyp, prevpos = typ, pos
    nblocks.append(('x', pos, 0))
    print(f"blocks w length out", nblocks[:6])
    return nblocks


def check_avg_time_per_unit(recd, chap):
    # get the chapter text and the length (ms) of all blocks
    # return the average time per text-unit
    unit_ct = 0
    blocks = G.block_ct[chap]
    for blkno in range(blocks):
        try:
            text = tt.get_block_text(recd, chap, blkno+1)
            units = get_units(text)
            unit_ct += units
        except AssertionError:
            pass
    dura_ct = 0

    for start, dura in G.block_dict.values():
        dura_ct += dura

    avg = dura_ct / unit_ct
    print(f'#blocks: {blocks}, avg time for block {dura_ct/blocks/1000 : 6.1f} sec')
    print(f'total time {int(dura_ct/1000)} sec, total units {unit_ct}, average time per unit {int(avg)} ms')
    return int(avg)


def show(recd, chap, block_dict, avg_tempo):
    block_numbers = sorted(block_dict.keys())

    time_seq =  []
    deviate  = []

    for blk in block_numbers:

        start, dura = block_dict[blk]

        # calculate the tempo (average time per unit) of one block
        try:
            text = tt.get_block_text(recd, chap, blk)  # may raise an assertion error
            units = get_units(text)
            tempo = int(dura / units)
        except AssertionError:  # tt.get_block_text() may raise this
            # the number of blocks may still be wrong, excess blocks will cause this error
            units = tempo = -1

        sec, ms = divmod(start, 1000)
        t = time.strftime('%H:%M:%S', time.gmtime(sec)) + f'.{ms:03d}'
        # calculate the deviation of the block tempo from the overall average tempo

        fact = 50
        dev = avg_tempo / tempo
        diff = int((1 - dev) * fact)
        if diff >= 0:
            diff = min(diff, 20)
            lstr = ' '* 20
            rstr = f"{'-'*diff:20}"
        else:
            diff = max(diff, -20)
            lstr = f"{'-'*-diff:>20}"
            rstr = ' '* 20

        time_seq.append((blk, t, tempo, units, dura, blk, dev, lstr +'|'+ rstr))

        # deviate.append((dev, t, tempo, units, dura, blk, dev))


    hdr = "  blk       start       dura  units   ms/t    dev"

    print(f"blocks in time sequence:\n")
    print(hdr)

    for _, t, tempo, units, dura, blk, dev , graph in sorted(time_seq):
        print(f"  {blk:3d}   {t}   {dura:5d}  {units:3d}    {tempo:4d} {dev:9.3f} {graph}")

    """
    select = int(len(deviate)/20 + 5)  # limit the lines, they must be processed top-down

    print(f"\nblocks with deviating time per text unit (avg={avg_tempo})")
    print(hdr)
    for dev, t, tempo, units, dura, blk, dev in sorted(deviate):
        if units < 1:
            continue
        if select > 0:
            select -= 1
            print(f"{blk:3d}   {t}   {dura:5d}  {units:3d}  {tempo:4d}  {dev:5.3f}")
    """
def get_units(text):
    vowels = "aiuAYW*"
    cv_string = ''.join(['v' if l in vowels else 'c' for l in text])
    unit_str = cv_string.replace("cv", "c")
    # print("unit string", unit_str)
    return len(unit_str)

main()
