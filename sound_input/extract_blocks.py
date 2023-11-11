# python3
""""""'''
    Split audio files (.wav, 24000 fps) of chapters (114) into blocks
    Blocks are stored as .wav files
    The block boundaries are stored in json files together with the necessary fixes
    We can get the fixed blocks from the FixBlocks class, which handles all the
    boundary management
    
    From the boundaries, get the finetuned place, where to cut
    Eliminate some noise
    Store the wav files for each block

'''
import numpy as np

from config import get_config
cfg = get_config()
from splib.sound_input_lib import FixBlocks
from splib.cute_dialog import start_dialog
from splib.sound_input_lib import get_amplitude_average
from matplotlib import pyplot as plt
import splib.toolbox as tbx
import splib.sound_tools as st


class G:
    fixblocks = None
    recd = ''

fpms = 24  # frames per millisecond
sil = 500  # milliseconds of silence (prefix, suffix)
extra_buffer = 1000
snip = 800  # ms size of sound in snippet
depth = 1500  # scan goes this deep into the block
charts = False
snippet = False

class dialog:
    recording = "hus1h"
    chapter = 0

    snippet = True
    winsize = 0
    max_ampl = 0
    iter_avg = 0

    layout = """
    title   Extract blocks from chapters
    text    recording  recording id, example: 'hus1h'
    text    chapter    koran chapter number or "*" (all)
    bool    snippet    create short sound snippets only
    bool    chart      draw the charts
    label   --- setting for boundary detection ---
    int     winsize   window size
    int     iter_avg  iterations of averaging
    int     max_ampl  maximum amplitude
    
    """

def main():
    if not start_dialog(__file__, dialog):
        return
    stt = tbx.Statistics()
    G.recd = recd = dialog.recording

    G.fixblocks = FixBlocks(cfg.work)
    G.fixblocks.load(recd)

    if dialog.chapter == "*":
        chap_seq = range(1, 115)
    else:
        chap_seq = [int(dialog.chapter)]

    for chap in chap_seq:
        print()
        fn = f"chap_{chap:03d}.wav"
        full = str(cfg.source / fn).format(recd=recd)
        fr, wav = st.read_wav_file(full)
        print(f"loaded chapter {chap}, len={len(wav)}")
        print()
        extra = np.full(extra_buffer * fpms, 0, dtype='int16')
        # avoid problems, when the wave has too little silence at the ends
        wav = np.concatenate((extra, wav, extra))

        # get an average over a small window size, to eliminate small peaks
        avg, total_loudness = get_amplitude_average(wav, dialog.winsize, dialog.iter_avg)
        max_ampl = dialog.max_ampl
        avglen = len(avg)

        print(f"length of wav {len(wav)}, avg {avglen}")
        print(f'block_loudness: {total_loudness}, modified max_ampl: {max_ampl}')

        blocktab = G.fixblocks.get_fixed_blocks(chap)
        #print(f"blocktab {chap}, {blocktab}")
        blkno = 0
        for typ, bstart, blen in blocktab:
            if typ != 'b':
                continue
            blkno += 1

            #if blkno < 235:
            #    continue

            bstart += extra_buffer  # compensate for the extra silence
            bend = bstart + blen   # bstart, bend refer to the position within the chapter
            print()
            print(f"boundaries blkno={blkno}, {bstart, bend} ms pos in chapter {chap}")

            # cut out the block audio data from the chapter, use the block boundaries
            fstart, fend = bstart * fpms, bend * fpms


            s, e = detect_boundaries(avg, max_ampl, bstart, bend)  # chapter coords in ms


            print(f"detected bounderies from {bstart, bend} to {s,e}")

            if s == -1 or e == -1:
                print(f"boundary detection failed for chap {chap}, blk {blkno}")
                continue

            frpos1, frpos4 = (s-sil) * fpms, (e+sil) * fpms

            if dialog.snippet:
                frpos2 = frpos1 + (sil+snip) * fpms
                frpos3 = frpos4 - (sil+snip) * fpms
                #print(f"snippet frames {frpos1, frpos2}  -  {frpos3, frpos4}")
                p1 = wav[frpos1:frpos2]
                p2 = wav[frpos3:frpos4]
                block = np.concatenate((p1, p2))
            else:
                block = wav[frpos1:frpos4]
                #print(f"cut out block at {frpos1, frpos4}")

            mspos1, mspos4 = s - sil, e + sil
            if dialog.snippet:
                mspos2 = mspos1 + sil+snip
                mspos3 = mspos4 - sil-snip
                p1 = avg[mspos1 : mspos2]
                p2 = avg[mspos3 : mspos4]
                avgbl = np.concatenate((p1, p2))
            else:
                avgbl = avg[mspos1:mspos4]
                #print(f"cut out average at {mspos1, mspos4}")

            print(f"boundaries blkno={blkno}, {s, e}, len={len(avg)}")
            blk_audio = edit_block_audio(block)

            if dialog.chart:
                editavg, _ = get_amplitude_average(blk_audio, dialog.winsize, dialog.iter_avg)
                print(f"editavg: blk={len(blk_audio)}, avg={len(editavg)}")
                # show final cuts in plot
                #avgbl[s-sil] = 1000
                #avg2[e+sil] = 1000
                plt.plot(avgbl)
                plt.plot(editavg)
                plt.show()

            st.write_wav_block(blk_audio, G.recd, chap, blkno, 24000)

    print(f"total time {stt.runtime()}")
    return


def edit_block_audio(block):
    # silence the boundaries of the block
    # the block was cut with {sil} ms of silence on both ends
    # --> just fade out both ends

    rmplen = 100*fpms
    sillen = sil*fpms
    linlen = sillen-rmplen

    # reduce noise to zero at the silent ends
    silvect = np.full(sillen, 0.0, dtype="float")
    silvect[linlen:sillen] = np.linspace(0, 1, rmplen)
    left_end = block[0:sillen].astype(float)
    left_end *= silvect
    block[0:sillen] = left_end.astype('int16')

    silvect = np.full(sillen, 0.0, dtype="float")
    silvect[0:rmplen] = np.linspace(1, 0, rmplen)
    right_end = block[-sillen:].astype(float)
    right_end *= silvect
    block[-sillen:] = right_end.astype('int16')

    return block

def detect_boundaries(avg, max_ampl, bst, bend):

    left_bd, right_bd = -1, -1

    # go from silence to sound
    #print(f"check left bndry {bst - sil, bst + depth}")
    for pos in range(bst - sil, bst + depth, +1):
        #print(f"check boundary  @{pos} a={avg[pos]}")
        if avg[pos] > max_ampl:
            left_bd = pos
            break
    #print(f"check right bndry {bend + sil, bend - depth}")
    for pos in range(bend + sil, bend - depth, -1):
        #print(f"check boundary  @{pos} a={avg[pos]}")
        if avg[pos] > max_ampl:
            right_bd = pos
            break
    return left_bd, right_bd,



main()
