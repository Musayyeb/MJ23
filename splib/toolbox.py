# python3
"""
    toolbox.py

    Collection of useful tools
"""
from config import get_config
cfg = get_config()

from collections import Counter
from enum import Enum
import multiprocessing as mp
import queue
import time
import datetime as DT
import json
import contextlib
import os
import math
import sys
import traceback
import logging as lg

class AttrDict(dict):
    # a dictionary class, which allows attribute access to all items
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
AD = AttrDict


def chap_blk_iterator(recd, chap_seq, blk_seq):
    # take chap_seq / blk_seq as text from the dialog window
    blk_counter = get_block_counters(recd) # a dictionary: key=chapno, value=blk_count

    if ':' in chap_seq:  # chap_no is either a single number or a range of n:m
        chp1, chp2 = chap_seq.split(':')
        chp_range = range(int(chp1), int(chp2))
    else:
        chp_range = range(int(chap_seq), int(chap_seq)+1)

    for chapno in chp_range:
        if not chapno in blk_counter:
            break

        if blk_seq == '*':
            blk_range = range(1, blk_counter[chapno]+1)
        elif ':' in blk_seq:  # blockno is either a single number or a range of n:m
            blk1, blk2 = blk_seq.split(':')
            blk_range = range(int(blk1), int(blk2))
        else:
            blk_range = range(int(blk_seq), int(blk_seq)+1)

        # this program processes n blocks in sequence
        for blkno in blk_range:

            yield chapno, blkno


class UJson():
    def __init__(self, fn):
        self.fn = fn
        if os.path.exists(fn):
            return
        with open(fn, mode="w") as fo:
            fo.write("[\n")

    def dump(self, obj):
        s = json.dumps(obj)
        with open(self.fn, mode="a") as fo:
            fo.write(s)
            fo.write(',\n')

    def load(self):
        with open(self.fn, mode='r') as fi:
            data = fi.read()
        data = data[:-2] + '\n]'
        obj = json.loads(data)
        return obj

class State(Enum):
    # status for the MPManager
    ok, first, final, failed = 0, 1, 2, 9
mpst = State

class MPManager():
    state = mpst

    def __init__(self, task_gen, worker_func, resp_hdl, wrkcount=4,
                 maxfail=7):
        self.task_gen = task_gen
        self.worker = worker_func
        self.resp_hdl = resp_hdl # generator style handler
        self.wrkcount = wrkcount
        self.maxfail = maxfail
        if wrkcount == 1:
            return

        self.taskq = mp.Queue()
        self.respq = mp.Queue()
        self.wrklist = []
        for wrkno in range(self.wrkcount):
            worker_id = f"WrkID-{wrkno+1:02d}"
            #w = MPWorker(self.taskq, self.respq, self.worker, worker_id)
            w = ShortMPWorker(self.run_multi, worker_id)
            w.start()
            self.wrklist.append(w)


    def run(self):
        lg.info(f"MPManager run {self.wrkcount} workers")
        if self.wrkcount == 1:
            self.run_single()
            return

        phase = 0
        pending = 0
        lomax = self.wrkcount * 4
        himax = lomax + 10
        failcount = 0
        active_wrk = self.wrkcount
        task_gen_time = 0.0
        task_gen_count = 0
        task_resp_time = 0.0
        task_resp_count = 0
        task_hdl_count = 0
        task_hdl_time = 0.0
        while True:
            # print(f"run loop pending: {pending}, more: {more_input}")

            if phase < 1 and pending < lomax:
                task_gen_start = time.time()
                for task_data in self.task_gen:
                    task_gen_count += 1
                    task_data.state=mpst.ok

                    self.taskq.put(task_data)
                    pending += 1
                    if pending > himax:
                        break
                else:
                    phase = 1  # go to finish
                task_gen_time += time.time() - task_gen_start

            if phase == 1:
                # task generator is exhausted or there is a problam
                # put termination "requests" for all workers
                for _ in range(self.wrkcount+1):
                    self.taskq.put(AD(state=mpst.final, task_id="terminate"))
                    pending += 1
                print("final requests are posted")
                phase = 2

            task_resp_count += 1
            task_resp_start = time.time()
            resp_data = self.get_response(waitt=0.5)  # wait for worker
            task_resp_time += time.time() - task_resp_start

            # get an immediate response or 0
            if resp_data == 0:
                continue

            pending -= 1

            task_hdl_count += 1
            task_hdl_start = time.time()
            self.resp_hdl.send(resp_data)   # wait for resp_hdlr
            task_hdl_time += time.time() - task_hdl_start

            if resp_data.state == mpst.final:
                active_wrk -= 1
                if active_wrk == 0:
                    print("finished all final tasks")
                    break
            if resp_data.state == mpst.failed:
                failcount += 1

                if failcount > self.maxfail:
                    phase = max(phase, 1)

        if task_gen_count:
            lg.info(f"task_gen count {task_gen_count}, time {task_gen_time:5.3f} "
                    f"avg: {task_gen_time / task_gen_count:1.6f}")
        if task_resp_count:
            lg.info(f"task_resp count {task_resp_count}, time {task_resp_time:5.3f} "
                    f"avg: {task_resp_time / task_resp_count:1.6f}")
        if task_hdl_count:
            lg.info(f"resp_hdl count {task_hdl_count}, time {task_hdl_time:5.3f} "
                    f"avg: {task_hdl_time / task_hdl_count:1.6f}")

    def get_response(self, waitt=1):
        try:
            return self.respq.get(timeout=waitt)
        except queue.Empty:
            return 0


    def run_single(self):
        # run single process without queuing of data
        print("MPM start Single Worker")
        wrkspc = AttrDict(state=mpst.first, worker_id="Single",
                          ct = Counter(), stats=Counter())
        failcount = 0

        for task_data in self.task_gen:
            task_data.state = mpst.ok

            resp_data = self.call_worker(wrkspc, task_data)

            self.resp_hdl.send(resp_data)

            if resp_data.state == mpst.final:  # termination request or failure
                print("MPM run_single ended because of state", resp_data.state)
                break

            if resp_data.state == mpst.failed:
                failcount += 1
                if failcount > self.maxfail:
                    break

        task_data = AD(state=mpst.final, task_id="terminate")
        resp_data = AD(stats=Counter(), msgs=[], task_id=task_data.task_id, state=task_data.state)

        self.worker(wrkspc, task_data, resp_data)
        self.resp_hdl.send(resp_data)
        print(f'MPM worker {wrkspc.worker_id}: ended')


    def run_multi(self, mpwork):
        print(f"MPM start MPWorker {mpwork.worker_id}")
        wrkspc = AttrDict(state=mpst.first,
                          worker_id=mpwork.worker_id, stats=Counter())

        while True:
            task_data = self.taskq.get() # blocking
            # task_data.state = mpst.ok  # not for multi-worker

            resp_data = self.call_worker(wrkspc, task_data)

            self.respq.put(resp_data)  # process output

            if resp_data.state == mpst.final:  # termination request or failure
                print("MPM run_mp ended because of state", resp_data.state)
                break

        print(f'MPM worker {mpwork.worker_id}: ended')


    def call_worker(self, wrkspc, task_data):

        resp_data = AD(task_id=task_data.task_id, state=task_data.state,
                       stats=Counter(), msgs=[])
        try:
            self.worker(wrkspc, task_data, resp_data)

        except Exception:
            s = f"\n\nWorker {wrkspc.worker_id} failed"
            print(s)
            print(task_data)
            resp_data.msgs.append(s)
            print("-" * 60)
            traceback.print_exc(file=sys.stdout)
            print("-" * 60)

            # let's assume, that the final request will not fail,
            # so its safe to set śtate as failed whithout loosing
            # track of finished workers
            resp_data.state = mpst.failed
        wrkspc.state = mpst.ok  # not first anymore
        return resp_data


class ShortMPWorker(mp.Process):
    def __init__(self, multi_worker, worker_id, ):
        super().__init__()
        self.worker = multi_worker # a worker function
        self.worker_id = worker_id

    def run(self):
        self.worker(self)



class Statistics(Counter):
    # this class supports the collection of runtime metrics

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.__dict__ = self

        self._starttime = time.time()
        self._log = []
        self.sowhat = "99"
        self.numseq = {}  # collect sequences
        self.numct  = {}  # count items

    def runtime(self, reset=False):
        # return a string for the elapsed time since start or reset
        dura = DT.timedelta(seconds=time.time()-self._starttime)
        if reset:
            self._starttime = time.time()
        return str(dura)

    def logmsg(self, text):
        self._log.append(text)

    def allmsg(self):
        return self._log

    def clrmsg(self):
        self._log = []

    def show(self):
        return '\n'.join(f"  {k} :  {v}"
                         for k,v in sorted(self.items())
                         if not k.startswith('_'))
    def show_log(self):
        for t in self._log:
            print(t)

    def collect(self, key, value):
        if not key in self.numseq:
            self.numseq[key] = []
            self.numct[key] = 0
        self.numseq[key].append(value)
        self.numct[key] += 1

    def keylist(self):
        return sorted(self.numseq.keys())

    def reset(self, key=None):
        if key:
            self.numseq[key] = []
            self.numct[key] = 0
        else:
            self.numseq = {}
            self.numct = {}

    def evaluate(self, key):
        # return average, variance and standard deviation
        # for a given collection of numbers
        seq, ct = self.numseq[key], self.numct[key]
        su = sum(seq)
        avg = su / ct
        var = sum([(n-avg)**2 for n in seq]) / ct
        stdd = math.sqrt(var)
        return AD(key=key, count=ct, sum=su, avg=avg, var=var, stdd=stdd)


def select_files(location, recd, chap="*", block="*", must_exist=True):
    # yield all files that match the given selection criteria
    # location is "blocks" (in the recording folder)   - or
    # any of the attribute name in the attribs folder
    if location == "blocks":
        path = str(cfg.blocks).format(recd=recd)
        pref, ext = 'b', 'wav'
    else:
        path = str(cfg.attribs / location).format(recd=recd)
        pref, ext = cfg.locations[location]
    if not os.path.exists(path):
        raise Exception("unknow folder location specified")

    print(f"select files: chap={chap}, block={block}")
    block_ct = get_block_counters(recd)
    schap = range(1,1000) if chap == '*' else [chap]
    print(f"resolved chap: {schap}")
    for fchap in sorted(block_ct.keys()):
        if fchap in schap:
            sblock = range(1, block_ct[fchap] + 1) if block == '*' else [block]
            print(f"resolved chap {fchap} block {sblock}")
            for fblk in range(1, block_ct[fchap] + 1):
                if fblk in sblock:

                    fn = f'{pref}_{fchap:03d}_{fblk:03d}.{ext}'
                    full = os.path.join(path, fn)
                    if os.path.exists(full):
                        yield fchap, fblk, fn, full
                    elif must_exist:
                        raise Exception(f"file not found {full}")
                    else:
                        continue # ignore missing files


def locate_data_files(location, recd, chap=None):
    # find the the files, which are there
    if not chap is None:
        chap = f"{chap:03d}"

    if location == "source":
        path = str(cfg.source).format(recd=recd)
    else:
        raise Exception("unknow location specified")

    files = os.listdir(path)
    for fn in sorted(files):
        if chap and not chap in fn:
            continue
        full = os.path.join(path, fn)
        yield fn, full

class RunToken():
    def __init__(self, runtok=None):
        self.runtok = cfg.work / runtok if runtok else None
        if runtok:
            print("To stop this program, delete ", self.runtok)
            with open(self.runtok, mode='w') as fo:
                pass  # create a run token. Delete the run token to stop the program

    def check_break(self):
        if self.runtok:
            return not os.path.exists(self.runtok)

    def clear(self):
        if self.runtok and os.path.exists(self.runtok):
            os.remove(self.runtok)


def now():
    # return a string for the current date and time
    curr = DT.datetime.now()
    return curr.strftime("%y.%m.%d %H:%M")


def get_block_counters(recd):
    # return a dictionary, which maps a chapter number to a block counter
    chaps = {}
    count_file = cfg.ldata / recd / "block_counter.txt"
    with open(count_file, mode='r') as fi:
        for line in fi:
            line = line.split('#')[0].strip()
            if line == '':
                continue
            chap, blk = line.split() # expect exactly 2
            chaps[int(chap)] = int(blk)

    return chaps

def test_stt():
    stt = Statistics()
    stt["hallo"] += 1
    stt["hallo"] += 1
    stt["hallo"] += 1
    stt.logmsg("halli")
    stt['juhu'] += 3
    stt['juhu'] += 3
    stt.logmsg("hallo")
    print(stt.show())
    print(stt.runtime())
    stt.show_log()

if __name__ == '__main__':
    print(f"this module {__file__} is for import")
    test_stt()
