# python3.6
''''''"""
    A graphical user interface to map "letters" to sound files.
    
    Letters represent sounds from the pronounciations of arabic text.
    Letters are also labels for the training of machine learning models.
    
    Soundfiles are the recorded readings of the koran text
    Soundfiles come in blocks (9000) with a length of mostly 10 to 30 seconds.
    
    To train the machine on sounds, segments of the sound must be labeled. This
    labeling of sounds is the task of manual mapping.
    
    The progam
    -- loads a specific block
        -- the block consists of the audio file (a .wav file)
           and a line of text (from the koran)
    -- displays the wave form of the audio
        -- the audio wave can be zoomed in and out
    -- displays the text line
        -- the text line is streched, so it matches the horizontal
           dimension of the audio wave display
    -- clicking on the audio diplay
        -- produces a audio signal of that part of the sound
        -- the sound duration is in the range of few seconds and can be adjusted
        -- the sound piece is faded in and out, so that the center of the
           selected sound can be identified well
    -- clicking on the text line
        -- associates the selected audio with the clicked letter
        -- adjusts the position of the letter on the x-axis
        
    A graphical user interface to help adjust the block boundaries of a koran chapter



"""

from config import get_config

cfg = get_config()
import splib.sound_tools as st
import splib.attrib_tools as attribs
import splib.text_tools as tt
from splib.toolbox import get_block_counters
from splib.cute_dialog import start_dialog
import splib.toolbox as tbx
from splib.sound_input_lib import FixBlocks, get_amplitude_average
from itertools import zip_longest
import time
import os
import numpy as np
from scipy import signal
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtCore import Qt, QTimer
from pyqtgraph.Point import Point
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QDialog, QStatusBar,
                             QDesktopWidget, QCheckBox, QVBoxLayout, QAction,
                             QHBoxLayout, QLineEdit, QGridLayout, QMainWindow)
from PyQt5.QtGui import QIcon, QFont
from functools import partial
import splib.project_db as pdb
import json
from splib.sockserv_tools import soundserv_request
import splib.colors as cor


class G:
    block_counters = {}

    block_ends = []
    block_dict = {}
    avg_win = 60  # ms window for sound average
    recd = 'hus9h'
    chap = 88
    blkno = 4
    wav = []  # wave array in 24000 fps

    dbman = None  # the db-manager oject
    conn = None   # connection for the current (recd) database

    label_mapping = None  # json data class - changed to db data class

    unmapped_positions = []
    koran_text = ''
    block_length = 0
    currpos = 0  # ms position of graphical cursor
    play = False  # True if playing
    timer = None
    time_interval = 75  # every n milliseconds
    removable_plot = []  # things that are plotted and eventually removed
    removable_text = []  # things that are plotted and eventually removed

    infinity = 9e12  # millisecond position long after the end of the audio

    # globally available gui elements
    cursor = None    # this is the vertcal line cursor in the plot
    recd_ed = None
    chap_ed = None
    blkno_ed = None     # edit field for the current block
    kortext_lbl = None  # label for the koran text
    timepos_lbl = None
    msg_lbl = None
    txtblks_ed = None
    audblks_ed = None

    plot = None  # plot widget

    plot_south = -3600  # keep space for working area
    plot_icon_bndry = (-400, -2000, -1200)
    plot_label_bndry = (-2000, -3500, -2200)



def main():
    pdb.db_connector(db_app)

def db_app(dbman, vdict=None):
    G.dbman = dbman


    app = QtGui.QApplication([])

    gui = QMainWindow()
    gui_setup(gui)

    G.timer = MyTimer()
    G.timer.register_tick_handler(handle_timer_intervals)
    G.timer.register_block_handler(handle_new_block_event)

    G.label_mapping = LabelMapping()
    # todo: initialize the values from previously saved settings (like dialog window)
    extern_set_recd(G.recd)
    extern_set_chap(G.chap)
    extern_set_blkno(G.blkno)

    gui.show()
    app.exec_()

    return


def gui_setup(gui):
    posx, posy, dimx, dimy = 1300, 200, 1000, 600
    gui.setGeometry(posx, posy, dimx, dimy)  # same same

    gui.setWindowTitle('Manually map positions of letters to sound')
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
    grid.addLayout(hbox, glin, 0, 1, 2)

    hbox.addWidget(QLabel('Recording'))

    G.recd_ed = ed = MyLineEdit()  # recording
    ed.setFont(med_font)
    ed.setMaxLength(10)
    ed.setMyFocusHandler(extern_set_recd)
    hbox.addWidget(ed)

    hbox.addStretch(1)

    hbox.addWidget(QLabel('Chapter'))

    G.chap_ed = ed = MyLineEdit()  # chapter
    ed.setAlignment(Qt.AlignCenter)
    ed.setFont(med_font)
    ed.setInputMask('999')
    ed.setMaxLength(5)
    ed.setMyFocusHandler(extern_set_chap)
    # todo: find the trick to have the data selected
    hbox.addWidget(ed)

    hbox = QHBoxLayout()
    grid.addLayout(hbox, glin, 3, 1, 2)

    btn = QPushButton('prev.Block (-1)')
    btn.setFont(med_font)
    btn.clicked.connect(partial(extern_chg_blk, -1))
    hbox.addWidget(btn)

    G.blkno_ed = ed = MyLineEdit()  # MyEdit
    ed.setAlignment(Qt.AlignCenter)
    ed.setFont(med_font)
    ed.setInputMask('999')
    ed.setMaxLength(5)  # todo: reduce size of input field
    ed.setMyFocusHandler(extern_set_blkno)
    hbox.addWidget(ed)

    btn = QPushButton('next Block (+1)')
    btn.setFont(med_font)
    btn.clicked.connect(partial(extern_chg_blk, +1))
    hbox.addWidget(btn)


    hbox.addStretch(1)

    btn = QPushButton('load')
    btn.setFont(med_font)
    btn.clicked.connect(load_block)
    hbox.addWidget(btn)

    hbox.addStretch(1)

    glin += 1

    graph = pg.GraphicsLayoutWidget()
    G.plot = plot = graph.addPlot(row=1, col=0)  # plotting area
    # react on mouse clicks on the graph window
    plot.scene().sigMouseClicked.connect(extern_plot_click)
    # add a cursor line
    G.cursor = vLine = pg.InfiniteLine(angle=90, movable=False)  # vertical line as a cursor
    plot.addItem(vLine, ignoreBounds=True)
    plot.setMouseEnabled(x=True, y=False)

    grid.addWidget(graph, glin, 0, 1, 5)  # pos=0,0,  use 1 row + 5 cols

    glin += 1

    btn = QPushButton('Play / Pause')
    btn.setFont(big_font)
    btn.clicked.connect(extern_play)
    grid.addWidget(btn, glin, 1, 1, 2)

    G.statbar = statbar = QStatusBar()
    gui.setStatusBar(statbar)
    G.msg_lbl = lbl = QLabel("fine")
    lbl.setFont(big_font)
    statbar.addWidget(lbl)

    # its possible to add text to the graph window
    # label = pg.LabelItem('what', justify='right') # a label inside the graph
    # graph.addItem(label)

    # end of gui_setup()


def extern_set_recd(recd):
    # after editing the recording field
    print(f"edit recd={recd}, prev {G.recd}")
    if recd != G.recd:   # chapter is changed
        G.plot.clear()
    dbref = G.dbman.connect(recd, "ml01")  # get a new reference for that database
    G.conn = dbref.conn   # the dbref has the logical connection
    G.recd = recd
    G.recd_ed.setText(recd)
    G.msg_lbl.setText('')


def extern_set_chap(chap_s):
    # after editing the chapter field (a string)
    chap = int(chap_s)
    print(f"edit chap={chap}, prev {G.chap}")
    if chap != G.chap:  # chapter is changed
        G.plot.clear()
    G.chap = chap
    G.chap_ed.setText(str(chap))
    G.msg_lbl.setText('')


def extern_chg_blk(dir):  # plus/minus buttons
    blkno = G.blkno + dir  # add +1 or -1
    extern_set_blkno(blkno)


def extern_set_blkno(blkno_s):  # blkno may be a string
    blkno = int(blkno_s)
    print(f"edit new block {blkno}, prev {G.blkno}")
    if blkno != G.blkno:  # block number is changed
        G.plot.clear()
    G.blkno = blkno
    G.blkno_ed.setText(str(blkno))
    G.msg_lbl.setText('')


def load_block(blkno):
    # action for the load button
    chap = G.chap
    recd = G.recd
    blkno = G.blkno

    print(f"load block: {chap}, {blkno}")

    # only here the validity of recd/chap/blkno settings are checked
    try:
        G.blk_counters = get_block_counters(recd)
    except FileNotFoundError:
        G.msg_lbl.setText(f'bad recording-id  {recd}')
        return

    if not chap in G.blk_counters:
        G.msg_lbl.setText(f'bad chapter {chap}')
        return

    block_fn = st.block_filename(recd, chap, blkno)
    try:
        fr, wav = st.read_wav_file(block_fn)
    except FileNotFoundError:
        G.msg_lbl.setText(f"block loading failed: {block_fn}")
        return
    freq_vect, pars_vect, rosa_vect = attribs.load_freq_ampl(recd, chap, blkno)
    freq_vect *= 40
    rosa_vect *= 15000
    pars_vect *= 50
    pars_vect[pars_vect < 0] = 0

    G.label_mapping.select_block(G.chap, G.blkno)
    G.audio_block = wav
    # print(len(wav), wav)

    soundserv_request(f'load {block_fn}')  # sound server loads the wav file

    avg, totavg = get_amplitude_average(wav, 30, 2, fpms=24)
    G.block_length = len(avg)

    G.koran_text = tt.get_block_text(G.recd, G.chap, blkno, spec='ml')

    plt = G.plot
    plt.clear()  # remove previous plots
    plt.setXRange(0, 5000, padding=0)  # 5 seconds?
    plt.setYRange(G.plot_south, totavg * 4, padding=0)  # adjust y-axis to average loudness

    avgx = np.linspace(0, len(avg), len(avg))
    plt.plot(avgx, avg, pen="w")
    freqx = np.linspace(0, len(avg), len(freq_vect))

    plt.plot(freqx, freq_vect, pen=cor.wheat)
    plt.plot(freqx, pars_vect, pen=cor.limegreen)
    plt.plot(freqx, rosa_vect, pen=cor.gold)

    # draw the lines for the map symbol area
    workhi, worklo, _ = G.plot_icon_bndry
    dimx = len(avg)
    plt.plot(np.full(dimx, worklo), pen='b')  # todo: use the np full method
    plt.plot(np.full(dimx, workhi), pen='b')

    reset_block_display()  # draw block/gap boundaries and fixes

    plt.addItem(G.cursor)


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

    workhi, worklo, _ = G.plot_label_bndry  # vertical position of the labels
    if worklo <= ypos <= workhi:  # click went into the fix area
        if G.currpos:  # only if a sound position is selected
            handle_map_label(xpos, G.currpos)  # add to list of mappings
            reset_block_display()
        return # the click went into the WORK area, no other action required
    workhi, worklo, _ = G.plot_icon_bndry  # vertical position of the map symbols
    if worklo <= ypos <= workhi:  # click went into the fix area
        handle_drop_label(xpos)  # remove from the list of mappings
        reset_block_display()

        return # the click went into the WORK area, no other action required

    # else place the cursor
    G.cursor.setPos(xpos)  # set position of graphical cursor
    G.currpos = xpos


def show_block(blkno, origin):
    print("show_block",blkno, origin)
    print("do what?")

def handle_map_label(label_px, audio_px):
    print(f"map label lblpos={label_px} audio_px={audio_px}")
    print(f"label positions: {G.unmapped_positions}")
    pxlo, pxhi = label_px-50, label_px+50
    for lndx, label, mspos in G.unmapped_positions:
        if mspos < pxlo:
            continue
        if mspos > pxhi:
            break
        print(f"map label '{label}' lndx={lndx} mspos={audio_px}")
        G.label_mapping.add_map(lndx, label, audio_px)
        G.currpos = 0
        G.cursor.setPos(0)  # set position of graphical cursor
        break

def handle_drop_label(px):
    pxlo, pxhi = px-20, px+20
    for lndx, label, mspos, rowid in get_mappings():
        if mspos < pxlo:
            continue
        if mspos > pxhi:
            break
        print(f"drop label '{label}' lndx={lndx}, pos={px}, rowid={rowid}")
        G.label_mapping.drop_map(rowid)
        break


def reset_block_display():
    # reset the plotted letters and positions, then redraw
    plt = G.plot
    chap = G.chap
    blkno = G.blkno

    # first: remove stored items
    for text in G.removable_text:
        plt.removeItem(text)

    for plit in G.removable_plot:
        plt.removeItem(plit)

    # then: clear stored items
    G.removable_plot = []
    G.removable_text = []

    # finally: plot the new items, again keep them stored, too
    labels_vect = get_labels_line()

    # print("labels_vect", labels_vect)

    for lbl_text in labels_vect:
        plt.addItem(lbl_text)
        G.removable_text.append(lbl_text)

    workhi, worklo, workmid = G.plot_icon_bndry

    for lndx, label, mspos, rowid in get_mappings():
        if label == 'X':
            return
        plit = plt.plot([mspos], [workmid], symbolBrush=(119, 172, 48),
                        symbol='d', symbolSize=20)
        G.removable_plot.append(plit)


def extern_key_event(evt):
    print(f'key_event: {evt}, {evt.key()}')
    # handle global keyboard keys
    if evt.key() == Qt.Key_Escape:
        terminate()

    # todo: add space key to start and stop audio playing

    if evt.key() == Qt.Key_Space:
        extern_play()

    if evt.key() == Qt.Key_Plus:
        extern_chg_blk(+1)
    if evt.key() == Qt.Key_Minus:
        extern_chg_blk(-1)


def terminate():
    # terminate the application
    if G.play:
        soundserv_request('stop')  # just in case sound is playing
    QApplication.instance().quit()


def get_labels_line():
    # create a plot vector for text labels

    labels_vect = []  # list of labels

    html = '<div style="color: #def; font-size: 14pt; text-align: center;">{}</div>'

    text = G.koran_text
    print(f"get labels line: {text}")
    _, _, label_ypos = G.plot_label_bndry
    map = get_mappings()
    G.unmapped_positions = []
    prev_lndx = 0
    prev_mspos = 300  # ms of silence at the start

    for lndx, label, mspos, rowid in map:
        print(f"label line {lndx, label, mspos, rowid}")
        if label != 'X':
            # these labels are already mapped
            lbl_text = pg.TextItem(html=html.format(label))
            lbl_text.setPos(mspos-25, label_ypos)  # xpos, ypos
            labels_vect.append(lbl_text)
            print(f"mapped lx={lndx}, '{label}' at {mspos} ")

        xdist = lndx - prev_lndx

        if xdist > 0:
            step = int((mspos - prev_mspos) / (xdist+1))
            miss_pos = prev_mspos
            print(f"prev_lndx {prev_lndx}, lndx {lndx} textlen {len(text)}")
            miss_ltrs = ''.join([text[p] for p in range(prev_lndx,lndx)])
            print(f"place missing lx={prev_lndx, lndx-1} at {miss_pos + step} - {miss_ltrs}")
            for miss_ndx in range(prev_lndx, lndx):

                miss_pos += step
                # there are letters left of the current label, which have to be placed
                label = text[miss_ndx]
                # print(f"place unmapped: {miss_ndx, label, miss_pos}")
                lbl_text = pg.TextItem(html=html.format(label))
                lbl_text.setPos(miss_pos-25, label_ypos)  # xpos, ypos
                labels_vect.append(lbl_text)
                G.unmapped_positions.append((miss_ndx, label, miss_pos))
        prev_lndx = lndx + 1
        prev_mspos = mspos


    return labels_vect


def get_mappings():
    # a generator, which return the next place, where a fix has to happen
    map = G.label_mapping.get_mmap_seq()
    for lndx, label, mspos, rowid in sorted(map):
        yield lndx, label, mspos, rowid

    yield len(G.koran_text), 'X', G.block_length-300, -1  # handle the silence on the right




def extern_play():
    # react to the play start/stop button

    pos = G.currpos

    wav = phon(pos)
    full = str(cfg.work / "snippet.wav")
    st.write_wav_file(wav, full, 24000)
    soundserv_request(f"play {full}")

    # while playing, the cursor should be moved forward to reflect
    # the current audio position
    #G.timer.start(pos)

def phon(pos):
    # pos in milliseconds
    # extract a  piece of audio from the audio block, which is left and right
    # of the current cursor position (ms)
    # fade in and out
    snip_size = 500
    lopos, hipos = pos - snip_size, pos + snip_size
    lofr, hifr = lopos*24, hipos*24
    print(f"prepare phon pos={pos}") # fr:{lofr, hifr}")

    snippet = G.audio_block[lofr:hifr].astype(float)
    snlen = len(snippet)
    gauss = signal.gaussian(snlen, snlen*0.25)
    clickpos = int(snlen/2)
    #print(gauss[:500], gauss[-500:])
    snippet *= gauss
    snippet = snippet.astype('int16')

    clickpos = int(snlen/2)
    for offs in (-5,-4,-3,3, 4, 5):
        snippet[clickpos+offs] = 10000
    for offs in (-2,-1,0,1,2):
        snippet[clickpos+offs] = 0

    return snippet

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
        self.pos = 0  # current position
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
        elaps = time.time() - self.wall_clock
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


class MyLineEdit(QLineEdit):
    # make subclass of QLineEdit, because there is no way to connect a focus event
    # this class calls the handler, only after the focus is lost (the change is complete)
    # the handler is called with the new value
    def setMyFocusHandler(self, handler):
        self.prev_val = self.text()
        self.my_focus_handler = handler

    def focusOutEvent(self, evt):  # when the focus is lost
        new = self.text()
        # print(f"focusOut event {evt}, text={new}")
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

class LabelMapping():
    '''
        this class does not need the recording - the recording is reflected
        in the db connection, which is set for the recording change code

        store already mapped labels
        for each block there is a key (ccc_bbb) and a list
        the items in this list are
            lndx   first label = 0
            letter (label)
            mspos  millisecond position in the audio
        the list must be kept sorted

    '''
    def __init__(self):
        self.recd = ""




    def _reset(self, recd):
        print('reset', recd)
        self.json_data = {'recording': recd,
                     'mappings': {},   # per chapter: a list of block/gap boundaries
                    }
        self.recd = recd
        return self.save()

    def select_block(self, chapno, blkno):

        sql = f"SELECT id, msoffs, label, lndx FROM 'manumap' where cbkey=?"
        cbkey = f"{int(chapno):03d}_{int(blkno):03d}"
        self.cbkey = cbkey
        csr = G.conn.execute(sql, (cbkey,))
        map = []
        for rowid, msoffs, label, lndx in csr:
            map.append((lndx, label, msoffs, rowid))
        self.map = sorted(map)


    def get_mmap_seq(self):
        return self.map


    def add_map(self, lndx, label, mspos):
        sql = f"""INSERT INTO 'manumap'
                    (cbkey, msoffs, label, lndx)
                    VALUES (?,?,?,?)"""
        csr = G.conn.execute(sql, (self.cbkey, mspos, label, lndx))
        rowid = csr.lastrowid
        G.conn.commit()  # commit each change individually

        self.map.append((lndx,label,mspos, rowid))
        self.map.sort()
        return

    def drop_map(self, rowid):
        sql = f"DELETE FROM 'manumap' WHERE id = ?"
        csr = G.conn.execute(sql, (rowid,))
        G.conn.commit()  # commit each change individually
        for lndx,label,mspos, row in self.map:
            if rowid == row:
                self.map.remove((lndx,label,mspos, row))
                break



class JsonLabelMapping():
    '''
        store already mapped labels
        for each block there is a key (ccc_bbb) and a list
        the items in this list are
            lndx   first label = 0
            letter (label)
            mspos  millisecond position in the audio
        the list must be kept sorted

    '''
    def __init__(self, path):
        self.path = path
        self.recd = ""
        self.json_mmap_fn = "manumap.json"
        self.json_data = {}


    def load(self, recd, reset=False):

        if recd == self.recd:
            print(f"'{recd}' already here")
            return True

        print('load', recd)
        json_full = self.path / recd / self.json_mmap_fn
        if os.path.exists(json_full):
            print("read json blocks")
            with open(json_full, mode='r') as fi:
                self.json_data = json.load(fi)

            self.recd = recd
            return True

        print(f"load {recd} failed")
        #if reset:
        return self._reset(recd)

        return False



    def _reset(self, recd):
        print('reset', recd)
        self.json_data = {'recording': recd,
                     'mappings': {},   # per chapter: a list of block/gap boundaries
                    }
        self.recd = recd
        return self.save()

    def save(self):
        json_full = self.path / self.recd / self.json_mmap_fn
        print('save', json_full)
        try:
            with open(json_full, mode='w') as fo:
                json.dump(self.json_data, fo)
        except FileNotFoundError as ex:
            print("saving json data failed", ex)
            return False

        return True

    def save_map(self):
        cbmaps = self.json_data['mappings']
        cbmaps[self.cbkey] = self.map
        self.save()

    def select_block(self, chap, blkno):
        # after selecting a block, later calls don't need chap and blkno
        self.cbkey = key = f'{chap:03d}_{blkno:03}'
        cbmaps = self.json_data['mappings']
        if not key in cbmaps:
            map = []
            cbmaps[key] = map
        self.map = sorted(cbmaps[key])


    def get_mmap_seq(self):
        return self.map


    def put_mmap_seq(self, map):
        self.map = sorted(map)
        self.save_map()
        return

    def add_map(self, lndx, label, mspos):
        self.map.append([lndx, label, mspos])
        self.map.sort()
        self.save_map()

    def drop_map(self, lndx, label, mspos):
        print(f"drop map {lndx} '{label}', {mspos}")
        print(f"map: {self.map}")
        for ndx, (x,l,p) in enumerate(self.map):
            if x == lndx:
                del self.map[ndx]
                self.save_map()
                return

    def x__init__(self, recd, chap, blkno):
        text = tt.get_block_text(G.recd, G.chap, blkno)



if __name__ == '__main__':
    main()
