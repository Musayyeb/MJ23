#python3
"""
    Extract machine learning attributes with the parselmouth (praat) tool
    Here we get the formants of the voice
    For the final prediction of all text, we need the full set of attributes
    The data volume will be excessive.
    write the data as numpy array data files. This is the fastest and most economic

"""
from config import get_config
cfg = get_config()
import splib.sound_tools as st
import splib.toolbox as tbx
AD = tbx.AD
import numpy as np
from splib.cute_dialog import start_dialog
import parselmouth
import logging as lg
import os
import time

mpst = tbx.MPManager.state  # shortcut for MultiProcessingManager states

class G:
    block_counters = {}
    runtoken = None

bundels = [['fmnts',
                ['fmnt_1', 'fmnt_2', 'fmnt_3','fmnt_4', 'fmnt_5','fmnt_6', 'fmnt_7']
           ]]

class dialog:
    recording = "hus1h"
    chapter = 100
    workers = 4

    layout = """
    title   Extract parselmouth formants from blocks
    text    recording  recording id, example: 'hus1h'
    text    chapter    koran chapter number or '*'
    label   -- for multiprocessing --
    int    workers     number of parallel workers
    """

def main():
    if not start_dialog(__file__, dialog):
        return

    global lg
    stt = tbx.Statistics()
    lg = stt.logmsg
    lg(f"start {dialog.recording} {dialog.chapter} workers: {dialog.workers}")

    G.runtoken = tbx.RunToken('extract_parsl.run_token')

    recd = dialog.recording
    G.block_counters = tbx.get_block_counters(recd)

    chap = dialog.chapter
    chap_sel = range(1,1000) if chap == '*' else [int(chap)]

    tasks = task_generator(stt, recd, chap_sel)

    resphd = response_handler(stt)

    resphd.send(None)

    mpm = tbx.MPManager(tasks, worker, resphd, wrkcount=dialog.workers)
    mpm.run()

    if G.runtoken.check_break():
        lg("terminated by runtime token")

    print("\nAll blocks done")

    print("runtime: ",stt.runtime())
    print(stt.show_log())
    print(stt.show())
    return

def task_generator(stt, recd, chap_sel):
    for chapno in chap_sel:
        if not chapno in G.block_counters:
            continue
        blkct = G.block_counters[chapno]
        # print(f"prepare {blkct} blocks for chap {chapno}")
        for blkno in range(1, 1 + blkct):

            if G.runtoken.check_break():
                break

            task_id = f"{chapno}:{blkno}"
            task_data = AD(task_id=task_id, recd=recd, chapno = chapno, blkno = blkno)
            stt['tasks'] += 1
            yield task_data
    print("task_generator ended")


def worker(wrkspc, task_data, resp_data):
    #time.sleep(1)
    m = resp_data.msgs.append  # shortcut for append function
    resp_data.prepro = False

    # if G.runtoken.check_break():
    #     task_data.state = mpst.final

    if task_data.state == mpst.final:  # this is the last call to this worker
        m(f"{wrkspc.worker_id} - termination call")
        resp_data.gstats = wrkspc.stats  # copy global statistics (wrkspc level)
        return

    if wrkspc.state == mpst.first:
        m(f"{wrkspc.worker_id} - First time call")
        wrkspc.chapno = '000'

    recd, chapno, blkno = task_data.recd, task_data.chapno, task_data.blkno

    if chapno != wrkspc.chapno:
        wrkspc.chapno = chapno
        print(f"{wrkspc.worker_id} working on chap {chapno}")  # write msg once per chapter

    # ### this is the real work part ###

    # m(f"{wrkspc.worker_id} - starts: {chapno}, {blkno}")
    # before processing this block, check, if it was processed before
    for bndl_id, bndl_attrs in bundels:
        fn = cfg.data / recd / "attribs" / ('pars_'+bndl_id) / f"{bndl_id}_{chapno:03d}_{blkno:03d}.npy"
        # m(f"check file {fn}: {os.path.exists(fn)}")
        if not os.path.exists(fn):
            break
    else:
        m(f"block is already there: {chapno}, {blkno}")
        resp_data.prepro = True
        return


    fr, wav = st.read_wav_block(recd, chapno, blkno)

    snd = parselmouth.Sound(wav, fr)

    fts = snd.to_formant_burg(time_step=0.005, max_number_of_formants=7,
                              maximum_formant=10000.0, window_length=0.025,
                              pre_emphasis_from=50.0)
    # print("fts", fts)
    bands = [[] for _ in range(7)]
    xs = fts.xs()
    for t in xs:
        for bx in range(7):
            bands[bx].append(fts.get_bandwidth_at_time(formant_number=bx+1, time=t))

    for band in bands:
        band.pop(0)   # this remove 1 element = 5ms from the begin ==> shift left

    # print(bands[3][500:600])
    for bndl_id, bndl_attrs in bundels:

        data = np.stack(bands)

        fn = cfg.data / recd / "attribs" / ('pars_'+bndl_id) / f"{bndl_id}_{chapno:03d}_{blkno:03d}.npy"
        m(f"write to {fn}")
        np.save(fn, data)
    m(f"{wrkspc.worker_id} - finished: {chapno}, {blkno}")


def response_handler(stt):
    while True:
        resp_data = yield 'resp'
        print(f"\nResponse received: {resp_data.task_id}\n")
        for msg in resp_data.msgs:
            print(f"    {msg}")
        stt['responses'] += 1
        if resp_data.prepro:
            stt['prev_processed'] += 1


def process_block(recd, chapno, blkno):
    pass

main()