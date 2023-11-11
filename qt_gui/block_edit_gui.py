# python3.6
''''''
"""
    A graphical user interface to help adjust the block boundaries of a koran chapter


    The application should have these functions
    - load and display an audio file (wav-format)
    - navigate (scroll) over the sound
    - play the sound from the current cursor position
    - zoom in and out of the display
    - maintain a list of blocks and gaps, as the identified so far
    - add and remove blocks and gaps
    - load and save the block mapping file

    - the main functions of the app are available as keyboard shortcuts

    for the gui this means there are several functional areas

    - menu
        -- load audio file (wav file) for a specific recording and chapter
        -- the current recording/chapter is always visible
        -- load and save the block mapping file, the filename relates to the audio file name

    - block display
        -- there is an input field, which allows to select a block
        -- there is a text field (read only), which displays the koran text for this block
        -- when the cursor moves, these fields are updated imediately
    - wav display
        -- show the wav amplitudes
            --- only mono,
            --- only positive half-wave (there is nothing new from the south)
            --- eventually add a loudness curve to the display
        -- adjust a zoom factor (plus/minus keys) (also +/- buttons on the side)
            --- the current wave curser keeps its position on the screen
                AND its position within the sound
            --- the increment is sqrt(2): press the key twice to douple/half the zoom ratio
        -- position the cursor in the wav display by mouse click
        -- move the wave cursor left and right (left and right cursor (arrow) keys)
            --- there is some magic scrolling, the cursor moves within 20-80% of the window,
                then the content is moved below the cursor. As audio is played forward,
                the magic scrolling position is at 30% of the wave screen
    - zoom display
        -- show the wave forms at the cursor position
            --- assume there is a constant zoom factor, which allows to 'see' the sound
                wave. the current cursor is always in the center of this window
        -- position the cursor in the zoom display, by mouse click
        -- move the zoom curser (actually move the wave image) by [alt]+ cursor key
           in small steps (also +/- buttons on the side
        -- Problem: zooming happens with the built-in mouse-scroll-wheel
            --- do we get a feedbacvk about the zomming that happens?

    - block/gap display
        -- there is a window, showing the blocks/gaps as a curve in synch with the wave display
        -- blocks show the block numbers

    - audio playing
        -- use the [p] key to start the audio playing, press [p] again to pause, then [p] to continue
        -- the cursor moves along with the audio output. if the audio output moves outside the
            current display, the wave is shifterd,so that the cursor is back in the visible
        -- the audio playing works with the sounddevice module
        -- Both pyqt and the sound device need a lot of cpu power. because of this the sound playing 
           is put into its own process (hope that works)
           

    - work with missing and extra blocks and gaps
        There are places, where the automatic block boundary detection fails.
        We need a data structure, where we keep track of corrections. This is a
        JSON file stored in the ldata for each recording
        -- define json data structur
        
                  
"""


from config import get_config
cfg = get_config()
import splib.sound_tools as st
import splib.text_tools as tt
from splib.toolbox import get_block_counters
from splib.cute_dialog import start_dialog
import splib.toolbox as tbx
from splib.sound_input_lib import FixBlocks, get_amplitude_average
from itertools import zip_longest
import time
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtCore import Qt, QTimer
from pyqtgraph.Point import Point
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QDialog,
                             QDesktopWidget, QCheckBox, QVBoxLayout, QAction,
                             QHBoxLayout, QLineEdit, QGridLayout, QMainWindow)
from PyQt5.QtGui import QIcon, QFont
from functools import partial
from splib.sockserv_tools import soundserv_request


class G:
    block_counters = {}

    block_ends = []
    block_dict = {}
    fixblocks = None
    avg_win = 60  # ms window for sound average
    recd = 'hus1h'
    chap = 1
    blockno = 1
    wav = []     # wave array in 24000 fps

    currpos = 0  # ms position of graphical cursor
    play = False  # True if playing
    timer = None
    time_interval = 75  # every n milliseconds
    removable_plot = []  # things that are plotted and eventually removed
    removable_text = []  # things that are plotted and eventually removed

    infinity = 9e12  # millisecond position long after the end of the audio

    # globally available gui elements
    curblk_ed = None  # edit field for the current block
    kortext_lbl = None  # label for the koran text
    timepos_lbl = None
    txtblks_ed = None
    audblks_ed = None
    plot = None   # plot widget

    plot_south = -4500   # keep space for working area
    plot_fix_area = (-2200, -4200)
    plot_fixes =  (-400, -2000)


def main():

    app = QtGui.QApplication([])

    gui = QMainWindow()
    gui_setup(gui)

    G.timer = MyTimer()
    G.timer.register_tick_handler(handle_timer_intervals)
    G.timer.register_block_handler(handle_new_block_event)


    G.fixblocks = FixBlocks(cfg.work)
    # todo: initialize the values from previously saved settings (like dialog window)
    extern_set_recd(G.recd)
    extern_set_chap(G.chap)


    gui.show()
    app.exec_()

    return


def gui_setup(gui):
    posx, posy, dimx, dimy = 1300, 200, 1000, 600
    gui.setGeometry(posx, posy, dimx, dimy)  # same same

    gui.setWindowTitle('Edit source files for mapping blocks')
    # create_menu(gui)

    gui.keyPressEvent = extern_key_event  # handle key-presses

    qw = QWidget()
    gui.setCentralWidget(qw)
    grid = QGridLayout()
    qw.setLayout(grid)


    # generate a layout, use a grid layout with 5 columns

    med_font = QFont("Arial", 12)
    big_font = QFont("Arial", 16)


    glin = 0  # grid line

    hbox = QHBoxLayout()
    grid.addLayout(hbox, glin,0, 1,1)

    hbox.addWidget(QLabel('Recording'))

    G.recd_ed = ed = MyLineEdit()   # recording
    ed.setFont(med_font)
    ed.setMaxLength(10)
    ed.setMyFocusHandler(extern_set_recd)
    hbox.addWidget(ed)

    hbox.addStretch(1)

    hbox = QHBoxLayout()
    grid.addLayout(hbox, glin,1, 1,2)

    btn = QPushButton('prev.Chapter (-1)')
    btn.setFont(med_font)
    btn.clicked.connect(partial(extern_chg_chap, -1))
    hbox.addWidget(btn)

    G.chap_ed = ed = MyLineEdit()   # chapter
    ed.setAlignment(Qt.AlignCenter)
    ed.setFont(med_font)
    ed.setInputMask('999')
    ed.setMaxLength(5)
    ed.setMyFocusHandler(extern_set_chap)
    hbox.addWidget(ed)

    btn = QPushButton('next Chapter (+1)')
    btn.setFont(med_font)
    btn.clicked.connect(partial(extern_chg_chap, +1))
    hbox.addWidget(btn)

    hbox = QHBoxLayout()
    grid.addLayout(hbox, glin,4,  1,1)

    hbox.addStretch(1)

    btn = QPushButton('load')
    btn.setFont(med_font)
    btn.clicked.connect(extern_load_chapter)
    hbox.addWidget(btn)

    hbox.addStretch(1)

    glin += 1
    hbox = QHBoxLayout()
    grid.addLayout(hbox, glin,1,  1,3)

    hbox.addWidget(QLabel('Text blocks -->'))

    G.txtblks_ed = ed = QLineEdit()
    ed.setReadOnly(True)
    ed.setAlignment(Qt.AlignCenter)
    ed.setFont(med_font)
    hbox.addWidget(ed)

    hbox.addStretch(1)

    G.audblks_ed = ed = QLineEdit()
    ed.setReadOnly(True)
    ed.setAlignment(Qt.AlignCenter)
    ed.setFont(med_font)
    hbox.addWidget(ed)

    hbox.addWidget(QLabel('<-- Audio blocks'))

    glin += 1

    graph = pg.GraphicsLayoutWidget()
    G.plot = plot = graph.addPlot(row=1, col=0)  # plotting area
    # react on mouse clicks on the graph window
    plot.scene().sigMouseClicked.connect(extern_plot_click)
    # add a cursor line
    G.cursor = vLine = pg.InfiniteLine(angle=90, movable=False)  # vertical line as a cursor
    plot.addItem(vLine, ignoreBounds=True)
    grid.addWidget(graph, glin,0, 1,5)  # pos=0,0,  use 1 row + 5 cols

    glin += 1
    hbox = QHBoxLayout()
    grid.addLayout(hbox, glin, 1, 1, 4)

    btn = QPushButton('prev.Block (-1)')
    btn.setFont(med_font)
    btn.clicked.connect(partial(extern_chg_blk, -1))
    hbox.addWidget(btn)

    G.curblk_ed = ed = MyLineEdit()   # MyEdit
    ed.setAlignment(Qt.AlignCenter)
    ed.setFont(med_font)
    ed.setInputMask('999')
    ed.setMaxLength(5)  # todo: reduce size of input field
    ed.setMyFocusHandler(extern_set_blk)
    hbox.addWidget(ed)

    btn = QPushButton('next Block (+1)')
    btn.setFont(med_font)
    btn.clicked.connect(partial(extern_chg_blk, +1))
    hbox.addWidget(btn)

    hbox.addStretch()

    G.timepos_lbl = lbl = QLabel('time')
    lbl.setFont(med_font)
    hbox.addWidget(lbl)

    hbox.addStretch()

    glin += 1
    hbox = QHBoxLayout()
    grid.addLayout(hbox, glin, 0, 1, 5)

    G.kortext_lbl = lbl = QLabel("koran text")
    lbl.setFixedSize(800, 80)
    lbl.setFont(med_font)
    lbl.setStyleSheet("border : 1px solid black; font-size: 16px;"
                      "font-family:Monospace;")
    lbl.setWordWrap(True)
    hbox.addWidget(lbl)  # pos=line,col=1,  use 2 row + 3 cols

    glin += 1

    btn = QPushButton('Play / Pause')
    btn.setFont(big_font)
    btn.clicked.connect(extern_play)
    grid.addWidget(btn, glin, 1, 1, 2)

    # its possible to add text to the graph window
    #label = pg.LabelItem('what', justify='right') # a label inside the graph
    #graph.addItem(label)

    # end of gui_setup()


def extern_set_recd(recd):
    print(f"edit recd={recd}, prev {G.recd}")
    try:
        G.blk_counters = get_block_counters(recd)
        #print("got block_counters:", G.blk_counters)
    except FileNotFoundError:
        print('bad recording-id', recd)
        return

    G.recd = recd
    G.recd_ed.setText(recd)
    G.block_counters = get_block_counters(recd)
    G.fixblocks.load(G.recd)

    # todo: still not ready: switch recd should reset everything


def extern_chg_chap(dir):  # plus/minus buttons
    nchap = G.chap + dir  # add +1 or -1
    print(f"btn: prepare new chap {nchap}")
    prepare_chapter(nchap)


def extern_set_chap(chap):
    nchap = int(chap)
    print(f"edit chap={nchap}, prev {G.chap}")
    prepare_chapter(nchap)


def prepare_chapter(chap):
    # collect infos about a new chapter and display it for the user
    try:
        G.block_counters[chap]
    except KeyError:
        print('bad chapter number', chap)
        return

    if chap != G.chap:
        # chapter is changed
        G.plot.clear()

    G.chap = chap
    G.chap_ed.setText(str(chap))
    # collect block info and block numbers
    if not G.block_counters:
        G.block_counters = get_block_counters(G.recd)
    G.txtblks_ed.setText(str(G.block_counters[chap]))

    read_block_boundaries(chap)
    G.audblks_ed.setText(str(len(G.block_ends)-2))
    #print(G.block_ends[:7])


def extern_load_chapter():
    # load everything, that is required for working on this chapter
    chap = G.chap
    recd = G.recd
    print(f"load chapter: {chap}")

    path = str(cfg.source).format(recd=recd)
    fn = f'chap_{chap:03d}.wav'
    full = os.path.join(path, fn)

    fr, wav = st.read_wav_file(full)
    # print(len(wav), wav)

    soundserv_request(f'load {full}')  # the sound server also has to load the wav

    avg, totavg = get_amplitude_average(wav, 10, 1, fpms=120)

    print('totavg',totavg)
    # blk_gap_vect, text_vect = get_labels_line(chap)

    plt= G.plot
    plt.clear()  # remove previous plots
    plt.setXRange(0, 45000, padding=0)  # 45 seconds?
    plt.setYRange(G.plot_south, totavg * 3, padding=0)  # adjust y-axis to average loudness
    plt.setMouseEnabled(x=True, y=False)

    avgx = np.linspace(0, len(avg)*5, len(avg))
    plt.plot(avgx, avg, pen="w")

    workhi, worklo = G.plot_fixes
    midfix = (workhi + worklo) / 2
    plotsize = len(avg) * 5
    plt.plot(np.array([workhi] * plotsize), color='cyan')  # todo: use the np full method
    plt.plot(np.array([worklo] * plotsize), color='cyan')

    reset_block_display()  # draw block/gap boundaries and fixes

    plt.addItem(G.cursor)

    G.blockno = 1
    show_block(1, 'input')



def extern_chg_blk(dir):  # plus/minus buttons
    nblk = G.blockno + dir  # add +1 or -1
    print(f"btn: go to new block {nblk}, prev {G.blockno}")
    show_block(nblk, 'input')


def extern_set_blk(newblk):  #
    nblk = int(newblk)
    print(f"edit new block {nblk}, prev {G.blockno}")
    show_block(nblk, "input")


def show_block(nblk, origin):
    # after the block number changed - update everything
    # origin is "input", "cursor" or "player"

    G.curblk_ed.setText(str(nblk))
    G.blockno = nblk


    # update the koran text
    # todo: find a better way to split long text lines (put a space into all mosaics)
    if nblk == 0:
        text = ""
    else:
        text = tt.get_block_text(G.recd, G.chap, nblk)
        text = insert_spaces(text)
    G.kortext_lbl.setText(text)

    if not nblk in G.block_dict:
        return

    start, dura = G.block_dict[nblk]
    print(f"block start position: {start}")

    # update clock time of the block-start
    sec, ms = divmod(start, 1000)
    t = time.strftime('%H:%M:%S', time.gmtime(sec)) + f'.{ms:03d}'
    G.timepos_lbl.setText(t)

    if not origin in ('player','cursor'):
        if G.play:
            G.timer.stop()
            soundserv_request('stop')
            G.play = False

    if not origin in ('cursor',):
        # out the cursor to the start of the new block
        G.cursor.setPos(start)
        G.currpos = start

        # position the block start to 25% of the window size
        # maintain the current zoom factor for the x-axis
        x_ax = G.plot.getAxis('bottom')
        lo, hi = x_ax.range
        span = hi - lo
        lo = start - span*0.25
        hi = start + span*0.75
        G.plot.setXRange(lo, hi, padding=0)


def extern_plot_click(evt):
    if evt.button() != 1:  # 1 => left click
        return

    plot = G.plot
    scene_coords = evt.scenePos()
    if not plot.sceneBoundingRect().contains(scene_coords):
        return

    mouse_point = plot.vb.mapSceneToView(scene_coords)
    xpos = int(mouse_point.x())
    ypos = int(mouse_point.y())
    # print(f'clicked plot X: {xpos}, Y: {ypos}')

    workhi, worklo = G.plot_fixes  # vertical position of the fix-area
    if worklo <= ypos <= workhi:  #click went into the fix area
        # click alone does not tell all
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:  # check for ctrl+enter
            mod = "mvrb"   # move right boundary to here
        elif modifiers == Qt.ShiftModifier:  # check for ctrl+enter
            mod = "mvlb"   # move left
        else:
            mod = "switch"  # switch between block and gap
        handle_fix_action(xpos, mod)  # add to (or remove from) list of fix actions
        read_block_boundaries(G.chap)
        reset_block_display()

        G.audblks_ed.setText(str(len(G.block_ends) - 2))

        # if the click went into the fix area, no other action is required
        return

    G.cursor.setPos(xpos)  # set position of graphical cursor
    G.currpos = xpos

    # update block number
    nblk = get_this_block(xpos)  # find blocknumber for current ms-position
    print(f'click goes to {xpos} block {nblk}, {G.block_ends[:6]}')
    show_block(nblk, 'cursor')

    if G.play:
        G.timer.stop()
        soundserv_request('stop')
        time.sleep(0.1)
        G.timer.start(xpos)
        soundserv_request(f'start {int(xpos)}')
        G.play = True


def handle_fix_action(px, mod):
    # add "fix-items" to (or remove from) the list of fixes
    print('do some fix action at', px, mod)
    # first check if we remove an action
    if remove_action(px):
        return  # remove action happened

    if mod == "mvrb" or mod == "mvlb":
        G.fixblocks.add_fix(G.chap, px, mod)
    else:
        assert mod == "switch"
        # check, if click is in a gap or in a block
        blkx = get_this_block(px)
        bpos, blen = G.block_dict[blkx]
        print(f"click target: {blkx}, pos:{bpos}, blen:{blen}")

        if bpos > px:  # then we are in a gap before that block
            # where is this gap ?
            prev_pos, prev_len = G.block_dict[blkx-1]
            gap_start = prev_pos + prev_len
            gap_pos = int((gap_start + bpos) / 2)
            print(f"gap pos={gap_pos} betw. {prev_pos, prev_len} and {bpos, blen}")

            # this gap is considered to not be a gap
            G.fixblocks.add_fix(G.chap, gap_pos, "nogap")
            return

        if bpos < px:  # then we are inside the given block
            # two options:
            if blen < 4000:  # it is a very small block, considerd "noise"
                G.fixblocks.add_fix(G.chap, px, "noblk")
            else:
                # assume there is some silence, which is considered a gap
                # no effort taken, to verify the silence or the real position
                G.fixblocks.add_fix(G.chap, px, "isgap")

    return True

def remove_action(px):
    x_ax = G.plot.getAxis('bottom')
    lo, hi = x_ax.range
    print(f"ax_range {lo} - {hi}")
    scale_fact = (hi-lo) / 70  # <-- right?
    pxlo, pxhi = px-scale_fact, px+scale_fact
    for pos, action in get_fixes(G.chap):
        if pos == G.infinity:
            return
        if pxlo < pos < pxhi:
            print(f"found fix action {action} @{pxlo}<{pos}<{pxhi} - remove")
            G.fixblocks.remove_fix(G.chap, pos, action)
            return True



def reset_block_display():
    # reset the plotted blocks, gaps and numbers, then redraw
    plt = G.plot
    chap = G.chap

    # first: remove stored items
    for text in G.removable_text:
        plt.removeItem(text)

    for plit in G.removable_plot:
        plt.removeItem(plit)

    # then: clear stored items
    G.removable_plot = []
    G.removable_text = []

    # finally: plot the new items, again keep them stored, too
    blk_gap_vect, text_vect = get_block_gap_line(G.chap)


    bgl = len(blk_gap_vect)
    bgx = np.linspace(0, bgl*5, bgl)
    plit = plt.plot(bgx, blk_gap_vect, pen="y")
    G.removable_plot.append(plit)

    for text in text_vect:
        plt.addItem(text)
        G.removable_text.append(text)

    workhi, worklo = G.plot_fixes
    midfix = (workhi + worklo) / 2

    for pos, item in get_fixes(chap):
        if item == "mvrb":
            symb, size = "t3", 16   # triangle pointing to the left
        elif item == "mvlb":
            symb, size = "t2", 16   # triangle pointing to the right
        else:
            symb, size = 'd', 20  # diamond
        plit = plt.plot([pos], [midfix], symbolBrush=(119, 172, 48), symbol=symb, symbolSize=size)
        G.removable_plot.append(plit)


def extern_key_event(evt):
    print(f'key_event: {evt}, {evt.key()}')
    # handle global keyboard keys
    if evt.key() == Qt.Key_Escape:
        terminate()

    if evt.key() == Qt.Key_Return:
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:  # check for ctrl+enter
            terminate()

    if evt.key() == Qt.Key_Plus:
        extern_chg_blk(+1)
    if evt.key() == Qt.Key_Minus:
        extern_chg_blk(-1)


def terminate():
    # terminate the application
    if G.play:
        soundserv_request('stop')  # just in case sound is playing
    QApplication.instance().quit()


def get_block_gap_line(chap):
    # create a plot vector for the blocks and gaps
    if len(G.block_ends) == 0:
        return [], []

    vect = []       # list of ndArrays
    text_vect = []  # list of block numbers

    blklvl, gaplvl = G.plot_fix_area  # y-positions of the plot area
    html = '<div style="color: #def; font-size: 14pt; text-align: center;">{}</div>'

    totlen = G.block_ends[-1][0]  # end of the audio
    vect = np.full(int(totlen/5), gaplvl)

    for blkx, (bpos, blen) in G.block_dict.items():
        text = pg.TextItem(html=html.format(blkx))
        tpos = bpos + blen / 3
        text.setPos(tpos, blklvl - 300)  # xpos, ypos
        text_vect.append(text)

        bpos, blen = int(bpos/5), int(blen/5)
        vect[bpos:bpos+blen] = blklvl


    return vect, text_vect


def get_fixes(chap):
    # a generator, which return the next place, where a fix has to happen
    fixes = G.fixblocks.get_fixes(chap)
    for pos, item in sorted(fixes):
        yield pos, item
    yield G.infinity, 'dummy'




def read_block_boundaries(chap):
    # the block boundaries are taken from a json file
    # this json file is managed by the FixBlocks Object

    block_dict = {}
    block_ends = []
    #blkpostab = G.fixblocks.get_raw_blocks(chap)
    #blocktab = blocks_w_lengths(blkpostab)

    blocktab = G.fixblocks.get_fixed_blocks(chap)
    # blocks_wo = [(t, p) for t,p,l in blocktab]  # blocktab withouth lengths
    # G.fixblocks.add_blocks(chap, blocks_wo)  # we NEVER update the blocks in Fixblocks

    blkx = 0
    for typ, bpos, blen in blocktab:
        if typ == 'b':
            blkx += 1
            block_dict[blkx] = bpos, blen
        if typ == 'g':
            if blkx == 0:
                block_ends.append((blen-50, blkx))
            else:
                block_ends.append((bpos, blkx))
        if typ == "x": # for the "dummy" block after the last gap
            block_ends.append((bpos, 999))

    G.block_dict = block_dict
    G.block_ends = block_ends
    #print(f"chap {chap} block dict", block_dict)
    print(f"block endings:", block_ends[-7:])

    return


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


def extern_play():
    # react to the play start/stop button

    if G.play:  # is playing now - stop playing
        soundserv_request('stop')
        G.play = False
        G.timer.stop()

    else:  # is not playing - start playing
        pos = G.currpos
        soundserv_request(f"start {int(pos)}")
        G.play = True

        # while playing, the cursor should be moved forward to reflect
        # the current audio position
        G.timer.start(pos)


class MyTimer:
    # this class does two things (smell!!!)
    # call a tick-handler for a running timer
    # call a block handler, if the block changest

    def __init__(self):
        self.timer = QTimer()
        self.timer.setInterval(G.time_interval)
        self.timer.timeout.connect(self.handle_tick)
        # the position is in ms from the start of the audio
        # so: position and time are equivalent
        self.pos = 0    # current position
        self.startpos = 0  # ms position of audio start
        self.wall_clock = 0  # set wall_clock time of start
        self.tick_handler = None
        self.block_handler = None

    def register_block_handler(self, handler):
        self.block_handler = handler
    def unregister_block_handler(self):
        self.block_handler = None

    def register_tick_handler(self, handler):
        self.tick_handler = handler
    def unregister_tick_handler(self):
        self.tick_handler = None

    def stop(self):
        self.timer.stop()

    def start(self, pos):
        self.wall_clock = time.time()
        self.startpos = pos
        self.pos = pos

        self.next_block = block_sequence(pos)  # setup a generator
        try:
            pos, blk = next(self.next_block)
            # this is position and number of the next block
            self.nexpos, self.nexblk = pos, blk
        except StopIteration:
            self.nexpos = 1e99  # infinity: we will never produce a 'next block' event

        self.timer.start()


    def handle_tick(self):
        elaps = time.time()-self.wall_clock
        # print(f"elapsed {elaps:6.3f}")
        self.pos = self.startpos + elaps * 1000
        if self.tick_handler:
            self.tick_handler(self.pos)

        if self.pos < self.nexpos:
            return

        try:
            pos, blk = next(self.next_block)
            self.nexpos, self.nexblk = pos, blk
        except StopIteration:
            self.nexpos = 1e99  # infinity: we will never produce a 'next block' event

        # we reached the postion of a new block
        if self.block_handler:
            self.block_handler(self.nexpos, self.nexblk)


def get_this_block(cursor):
    for pos, blkx in G.block_ends:
        if pos > cursor:
            return blkx
    return 0


def block_sequence(start_pos):
    for pos, blkx in G.block_ends:
        # for a given start positon yield all positions after that
        if pos > start_pos:
            print(f'block_sequence {pos}: {blkx}')
            yield pos, blkx


def handle_timer_intervals(pos):
    # react to the timer ticks
    G.currpos = pos
    G.cursor.setPos(G.currpos)  # Update the positon of the graph cursor


def handle_new_block_event(pos, blk):
    # this is called from the G.timer when a block boundary is reached
    print(f'player got new block {blk} at pos {pos}')
    if blk <= G.block_counters[G.chap]:
        show_block(blk, 'player')


'''def create_menu(gui):
    menu = gui.menuBar()
    file_menu = menu.addMenu('File')
    edit_menu = menu.addMenu('Edit')
    help_menu = menu.addMenu('Help')

    load_m = QAction(QIcon(), 'Load Sound', gui)
    # load_m.triggered.connect(partial(load_sound_dlg, gui))
    file_menu.addAction(load_m)

    exit_m = QAction(QIcon(), 'Exit', gui)  # QIcon('exit24.png')
    exit_m.setShortcut('Ctrl+Q')
    exit_m.setStatusTip('Exit application')
    exit_m.triggered.connect(gui.close)
    file_menu.addAction(exit_m)
''' # create menu


class MyLineEdit(QLineEdit):
    # make subclass of QLineEdit, because there is no way to connect a focus event
    # this class calls the handler, only after the focus is lost (the change is complete)
    # the handler is called with the new value
    def setMyFocusHandler(self, handler):
        self.prev_val = self.text()
        self.my_focus_handler = handler

    def focusOutEvent(self, evt):  # when the focus is lost
        new = self.text()
        print(f"focusOut event {evt}, text={new}")
        if new != self.prev_val:  # may crash, if no handler was connected
            self.prev_val = new
            self.my_focus_handler(new)  # call the handler

        super(MyLineEdit, self).focusOutEvent(evt)

def insert_spaces(text):
    newtext = []
    for l1, l2 in zip(text, text[1:]):
        newtext.append(l1)
        if cons(l1) and cons(l2):
            newtext.append(' ')
    newtext.append(l2)
    return "".join(newtext)

vowels = "aiuANYW*."

def cons(ch):
    return False if ch in vowels else True

if __name__ == '__main__':
    main()
