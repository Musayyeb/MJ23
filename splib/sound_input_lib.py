# python5
"""
    Collection of functions in the sound input part of the project
"""
import numpy as np
import splib.sound_tools as st
import os
import json

def get_amplitude_average(wav, winsize, iterations=1, fpms=24):
    abswav = np.absolute(wav)  # fold negative side to positive
    # converted data to 1 ms frames: window size IS in milliseconds
    total_loundess = int(np.average(abswav))
    # reduce the sample rate (frame rate), by averaging existing samples
    resolution = fpms # frames per ms
    sample_count = len(abswav) // resolution
    ms_len = sample_count * resolution  # the last frames may be lost
    # regroup (reshape) the frames to one value (per grp)
    data = abswav[:ms_len].reshape(sample_count, resolution)
    # calculate the average for the values for each sample
    msdata = np.average(data, axis=1)
    msdata = msdata.astype(int)
    # calculate the average ms values over a given window size
    msdata = st.running_mean(msdata, winsize, iterations)
    avg = msdata.astype(int)
    return avg, total_loundess



class FixBlocks:
    infinity = 9e12  # millisecond position long after the end of the audio

    def __init__(self, path):
        self.path = path
        self.recd = ""
        self.json_blk_fn = "block_boundaries.json"
        self.json_fix_fn = "fix_boundaries.json"
        self.json_data = {}


    def load(self, recd, reset=False):
        if recd == self.recd:
            print(f"'{recd}' already here")
            return True

        print('load', recd)
        json_full = self.path / recd / self.json_blk_fn
        if os.path.exists(json_full):
            print("read json blocks")
            with open(json_full, mode='r') as fi:
                self.json_blk_data = json.load(fi)

        json_full = self.path / recd / self.json_fix_fn
        if os.path.exists(json_full):
            print("read json fixes")
            with open(json_full, mode='r') as fi:
                self.json_fix_data = json.load(fi)

            self.recd = recd
            return True

        print(f"load {recd} failed")
        if reset:
            return self._reset(recd)

        return False


    def _reset(self, recd):
        print('reset', recd)
        self.json_blk_data = {'recording': recd,
                     'blk_boundaries': {},   # per chapter: a list of block/gap boundaries
                    }
        self.json_fix_data = {'recording': recd,
                     'fix_chapters': {},   # per chapter: a list of fixes
                    }
        self.recd = recd
        return self.save()


    def get_root(self):
        return self.json_data


    def save(self):
        json_full = self.path / self.recd / self.json_blk_fn
        print('save', json_full)
        try:
            with open(json_full, mode='w') as fo:
                json.dump(self.json_blk_data, fo)
        except FileNotFoundError as ex:
            print("saving json data failed", ex)
            return False

        json_full = self.path / self.recd / self.json_fix_fn
        print('save', json_full)
        try:
            with open(json_full, mode='w') as fo:
                json.dump(self.json_fix_data, fo)
        except FileNotFoundError as ex:
            print("saving json data failed", ex)
            return False
        return True



    def check_boundaries(self,chap, mslo, mshi):
        # check. if there is any "fix item" for the chapter in the given millisecond range
        print('check')
        cd = self.json_fix_data['fix_chapters']
        # check, what is in the bounddaries data
        key = f'chap_{chap:03d}'
        if not key in cd:
            return

        for pos, item in cd[key]:
            if mslo <= pos < mshi:
                yield pos, item


    def add_fix(self, chap, pos, item):
        print('add')
        cd = self.json_fix_data['fix_chapters']
        key = f'chap_{chap:03d}'
        if not key in cd:
            cd[key] = []
        items = cd[key]
        items.append([pos, item])
        print('     current',items)
        cd[key] = sorted(items)

        self.save()  # always immediately save


    def remove_fix(self, chap, pos, item):
        cd = self.json_fix_data['fix_chapters']
        key = f'chap_{chap:03d}'
        if not key in cd:
            print('nothing to remove')
            return
        items = cd[key]
        try:
            items.remove([pos, item])
        except ValueError:
            print('nothing to remove')
            return

        self.save()  # always immediately save


    def get_fixes(self, chap):
        # return a list of fixes for a chapter, list may be empty
        cd = self.json_fix_data['fix_chapters']
        key = f'chap_{chap:03d}'
        if key in cd:
            print(f'get {chap} fixes ({len(cd[key])}) ')
            return cd[key]
        else:
            return []


    def add_blocks(self, chap, boundaries):
        # always overwrite the lists
        bb = self.json_blk_data['blk_boundaries']
        key = f'chap_{chap:03d}'
        bb[key] = boundaries

    def get_raw_blocks(self, chap):
        bb = self.json_blk_data['blk_boundaries']
        key = f'chap_{chap:03d}'
        if not key in bb:
            print('block boundary data is missing for', key)
            return []
        return bb[key]

    def get_fixed_blocks(self, chap):
        blkpostab = self.get_raw_blocks(chap)
        blocktab = self.blocks_w_lengths(blkpostab)  # adds type (b/g) and length
        print(f"get fixed blocks: in count={len(blocktab)}")
        blocktab = self.apply_fixes(chap, blocktab)
        print(f"get fixed blocks: out count={len(blocktab)}")
        return blocktab

    def blocks_w_lengths(self, blocks):
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


    def fixes_gen(self, chap):
        # a generator, which return the next place, where a fix has to happen
        fixes = self.get_fixes(chap)
        for pos, item in sorted(fixes):
            yield pos, item
        yield self.__class__.infinity, 'dummy'



    def apply_fixes(self, chap, blocktab):
        # blocktab comes with (typ, pos, len)
        # return a new block tab with changed block/gap boundaries

        fixtab = self.fixes_gen(chap)

        while True:

            newblocks = []

            fixpos, fixaction = next(fixtab)
            print(f"Fix Action to apply: {fixpos}  {fixaction}")
            # print('blocktab:', blocktab)

            if fixpos == self.__class__.infinity:
                newblocks = blocktab
                break

            for ndx, ((t1, p1, l1), (t2, p2, l2)) in enumerate(zip(blocktab, blocktab[1:])):

                # t, p, l = type (b/g), pos, length
                if p2 < fixpos:
                    newblocks.append((t1, p1, l1))
                    continue

                # we detected a fix for item 1
                if fixaction == 'mvlb':
                    # we are already to the right of p1
                    print(f"action {fixaction} at {fixpos} found: {t1,p1,l2} - {t2,p2,l2}")
                    # we are inside a block or a gap (t1)
                    # adjust the p1 and l1, then also adjust length of the prev item
                    diff = fixpos - p1
                    p1 = fixpos
                    l1 -= diff
                    t0, p0, l0 = newblocks[-1]
                    l0 += diff
                    newblocks[-1] =((t0, p0, l0))
                    newblocks.append((t1, p1, l1))

                    newblocks.extend(blocktab[ndx + 1:])
                    break

                if fixaction == 'mvrb':
                    # we are already to the right of p1
                    print(f"action {fixaction} at {fixpos} found: {t1,p1,l2} - {t2,p2,l2}")

                    diff = p2 - fixpos
                    p2 = fixpos
                    l2 += diff
                    l1 -= diff
                    newblocks.append((t1, p1, l1))
                    newblocks.append((t2, p2, l2))

                    newblocks.extend(blocktab[ndx + 2:])
                    break



                elif fixaction in ('nogap', 'noblk'):
                    # actally the same action, independent of the type
                    t0, p0, l0 = newblocks[-1]
                    newblocks[-1] = ((t0, p0, l0 + l1 + l2))

                    newblocks.extend(blocktab[ndx + 2:])  # the current t1 is skipped
                    break

                elif fixaction == 'isgap':
                    if t1 == 'b':
                        # we could go into the audio, and find the silent part
                        # for now I (Hans) am too lazy - just insert a short gap
                        hgap = 80  # half gap
                        lnew = fixpos - p1 - hgap
                        newblocks.append((t1, p1, lnew))
                        newblocks.append(('g', fixpos - hgap, hgap + hgap))
                        lnew = p2 - fixpos + hgap
                        newblocks.append(('b', fixpos + hgap, lnew))

                        newblocks.extend(blocktab[ndx + 1:])
                        break


            else:
                # some fix action is not processed (???)
                newblocks.append((t2, p2, l2))

            blocktab = newblocks
            # print('blocktab:', blocktab)

        return newblocks

