# python3
"""
    A cute dialog is made with QT
"""
import config as glbcfg
cfg = glbcfg.get_config()
import sys
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QDesktopWidget, QCheckBox, QVBoxLayout,
                             QHBoxLayout, QLineEdit, QGridLayout)
from splib import toolbox as tbx
AD = tbx.AttrDict
from functools import partial
import pickle
import time
import os

class G:
    pass
    lbl_text = "duumiaa"
    setting_fn = 'saved_config.pickle'  # all the settings go into one single pickle file
    setting_dict = None



def start_dialog(origin, dlg, verbose=True):
    # dlg is a class like object

    interface = parse_interface(dlg.layout)

    settings = SettingsRepo(origin)
    data = AD(settings.read())
    #print("saved settings:", data)

    valdict = handle_saved_and_defaults(data, interface, dlg)
    valdict.start_app = False
    #print(f"valdict before: {valdict}")

    start_window(interface, valdict)

    handle_dialog_input(valdict, dlg)
    if verbose:
        print(f"start_dialog input: {valdict}")

    #print(f"valdict after: {valdict}")
    settings.write(valdict)

    return valdict.start_app

def start_window(interface, valdict):

    #intern = parse_interface(interface, valdict)
    app = QApplication(sys.argv)
    w = QWidget()  # no parent: a root window
    w.setWindowTitle("PyQt5")
    posx, posy, dimx, dimy = 300, 200, 250, 250
    # w.resize(dimx, dimy)
    # w.move(posx, posy)
    # center(w)
    w.setGeometry(posx, posy, dimx, dimy)  # same same

    def bool_event(ndx, state):
        #print("bool_event", ndx, state)
        varname = interface[ndx].name
        valdict[varname] = True if state == Qt.Checked else False

    def text_event(ndx, text):
        #print("text_event", ndx, text)
        wdict = interface[ndx]
        varname = wdict.name
        if wdict.data == 'text':
            valdict[varname] = text
        elif wdict.data == 'int':
            try:
                valdict[varname] = int(text) if text else 0
            except ValueError:
                print("cute_dialog: wrong input")
        elif wdict.data == 'float':
            try:
                valdict[varname] = float(text) if text else 0.0
            except ValueError:
                print("cute_dialog: wrong input")
                # better change the color of the input field

    def start_event(e):
        # print("button event", e)
        valdict.start_app = True
        QApplication.instance().quit()

    def key_event(e):
        #print(f"key.event {e.key()}")
        #print(f"key modifier {e.modifiers()}")
        #print("return", Qt.Key_Return)

        if e.key() == Qt.Key_Escape:
            valdict.start_app = False
            QApplication.instance().quit()
            # w.close()  # also works??

        if e.key() == Qt.Key_Return: # enter does not work for the window
            modifiers = QApplication.keyboardModifiers()
            if modifiers == Qt.ControlModifier:
                valdict.start_app = True
                QApplication.instance().quit()

    #vbox = QVBoxLayout()
    grid = QGridLayout()
    for ndx, item in enumerate(interface):
        # print(ndx, item)
        if item.wtype == 'title':
            w.setWindowTitle(item.wtext)

        elif item.wtype == 'label':
            lbl = QLabel()
            lbl.setText(item.wtext)
            #vbox.addWidget(lbl)
            grid.addWidget(lbl, ndx, 0, 1, 2)

        elif item.wtype == 'check':
            cb = QCheckBox(item.wtext)
            value = valdict[item.name]
            #print(f"checked {item.name} / {value}")
            cb.setCheckState(Qt.Checked if value else Qt.Unchecked)

            this_bool = partial(bool_event, ndx)
            cb.stateChanged.connect(this_bool)
            grid.addWidget(cb, ndx, 0, 1, 2)

        elif item.wtype == 'text':
            this_text = partial(text_event, ndx)
            le = QLineEdit()
            le.setMaxLength(25)
            if item.data == 'text':
                le.setAlignment(Qt.AlignLeft)
            else:
                le.setAlignment(Qt.AlignRight)
            #le.setPlaceholderText(str(item.val))
            value = valdict[item.name]
            le.setText(str(value))
            le.textChanged.connect(this_text)
            # le.textEdited.connect(this_text) # no obvious difference
            lbl = QLabel()
            lbl.setText(item.wtext)
            # vbox.addWidget(le)
            grid.addWidget(lbl, ndx, 0)
            grid.addWidget(le, ndx, 1)

    grid.addWidget(QLabel(''), ndx+1, 0)
    #vbox.addStretch(1)
    #hbox = QHBoxLayout()
    quit_btn = QPushButton('Cancel')
    quit_btn.clicked.connect(QApplication.instance().quit)
    quit_btn.resize(quit_btn.sizeHint())
    #hbox.addWidget(quit_btn)
    grid.addWidget(quit_btn, ndx+2, 0)

    run_btn = QPushButton('Start')
    run_btn.clicked.connect(start_event)
    run_btn.resize(quit_btn.sizeHint())
    #hbox.addWidget(quit_btn)
    grid.addWidget(run_btn, ndx+2, 1)

    #vbox.addLayout(hbox)
    #w.setLayout(vbox)
    w.setLayout(grid)

    w.keyPressEvent = key_event

    w.show()  # the empty frame appears immediately
    app.exec_()

    return valdict


def parse_interface(itf_text):
    intern = []

    for line in itf_text.splitlines():
        line = line.split('#')[0].strip()
        if not line:
            continue
        tup = line.split()
        linetype = tup[0]
        # associate a widget type for every line type
        if linetype == "title":
            wtext = ' '.join(tup[1:])
            intern.append(AD(wtype='title', wtext=wtext))
        elif linetype == "label":
            wtext = ' '.join(tup[1:])
            intern.append(AD(wtype='label', wtext=wtext))
        elif linetype == "bool":
            varname = tup[1]
            wtext = ' '.join(tup[2:])
            intern.append(AD(wtype='check', wtext=wtext, name=varname))
        elif linetype == "text":
            varname = tup[1]
            wtext = ' '.join(tup[2:])
            intern.append(AD(wtype='text', data='text', wtext=wtext, name=varname))
        elif linetype == "int":
            varname = tup[1]
            wtext = ' '.join(tup[2:])
            intern.append(AD(wtype='text', data='int', wtext=wtext, name=varname))
        elif linetype == "float":
            varname = tup[1]
            wtext = ' '.join(tup[2:])
            intern.append(AD(wtype='text', data='float', wtext=wtext, name=varname))

        else:
            raise Exception(f"bad line type {linetype}")

    intern.append(AD(wtype='check', wtext='save settings', name='saveset'))

    return intern


def handle_saved_and_defaults(saved, interface, dialog):
    # the interface determins, which values are displayed
    # the saved values c priority, when missing may
    # if these values are in the dialog object, they initialize/overwrite
    # the values in the saved settings
    # print(f"dialog (AD): {dialog}")
    settings = AD(saved)
    referred = set()
    for tok in interface:
        if "name" in tok:
            name = tok.name
            referred.add(name)
            if not name in settings:
                if hasattr(dialog, name):
                    settings[name] = getattr(dialog, name)
                else:
                    settings[name] = "0"

    # remove names that are not in the interface from the setting "history"
    new_settings = AD({k:v for k,v in settings.items() if k in referred})
    return new_settings


def handle_dialog_input(valdict, dialog):
    # the interface determins, which values are displayed
    # if these values are in the dialog object, they initialize/overwrite
    # the values in the saved settings
    # print(f"dialog (AD): {dialog}")
    for key,val in valdict.items():
        setattr(dialog,key, val)

    return

class SettingsRepo:
    def __init__(self, origin):
        self.origin = origin
        self.alldata = {}

        self.settings_fn = cfg.work / G.setting_fn
        if os.path.exists(self.settings_fn):
            with open(self.settings_fn, mode='rb') as pfn:
                self.alldata = pickle.load(pfn)
        if not origin in self.alldata:
            # there is always an entry for the current origin
            self.alldata[origin] = {"saveset" : True}

    def read(self):
        return self.alldata[self.origin]

    def write(self, settings):
        # this is an update of the alldata[origin]
        # if the saveset flag is on, all values are stored
        # else only the saveset flag is updated
        save_flag = True if settings and settings.saveset else False
        #print(f"write settings, flag={save_flag}")
        if save_flag:
            self.alldata[self.origin] = settings
        else:
            self.alldata[self.origin]["saveset"] = False
        #print(f"write pickle {self.alldata[self.origin]}")
        with open(self.settings_fn, mode='wb') as pfn:
            pickle.dump(self.alldata, pfn)
        return

if __name__ == '__main__':
    print('this is a module for import')

    print("running this module will reset the settings of a module")
    module = 'mapping/plot_prediction_spot.py'

    fn = cfg.work / G.setting_fn
    if os.path.exists(fn):
        with open(fn, mode='rb') as pfn:
            alldata = pickle.load(pfn)
        for k in sorted(alldata.keys()):
            print(k)
            if module in k:
                del alldata[k]
        with open(fn, mode='wb') as pfn:
            pickle.dump(alldata, pfn)
