# python3
"""
    Configuration settings for the phonemics project
    usage:
        from config import get_config
        cfg = get_config()
        # cfg is an attribute dictionary
"""
from pathlib import Path
import os
import local_config as lcfg

class AttrDict(dict):
    # a dictionary class, which allows attribute access to all items
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
AD = AttrDict

lpath = Path(os.path.dirname(__file__))



def get_config():
    cfg = AD()
    cfg.work = lpath / "workfiles"  # project specific non-shared data
    cfg.ldata = lpath / "ldata"   # project specific shared data

    cfg.data = lcfg.data    # root for the data
    cfg.recording = lcfg.data / "{recd}"  # recording specific data
    cfg.attribs = lcfg.data / "{recd}" / "attribs"
    cfg.blocks = lcfg.data / "{recd}" / "blocks"
    cfg.source = lcfg.data / "{recd}" / "source"
    cfg.probs = lcfg.data / "{recd}" / "probs"   # probability matrices
    cfg.pred_cache = lcfg.data / "{recd}" / "pred_cache"

    cfg.audio_path = lcfg.audio_path

    cfg.block_fn = "s_{chap:03d}_{blk}"
    cfg.recording_fn = "chap_{chap:03d}.wav"

    cfg.locations = locations = dict(
        formants = ('formants', 'npy'),
        freq_ampl=('freq_ampl', 'npy'),
        zcr=('zcr', 'npy'),
    )
    cfg.framerate = 24000
    return cfg