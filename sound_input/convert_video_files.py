# python3
"""
    The koran audio input is downloaded in the form of video files from youtube
    The tool to download is the firefox DownloadHelper plugin
    The video files come with various extension (mkv, webm, ...)
    Luckily ffmpeg does the magic and converts all(?) formats automatically
    The output files are in wav format with 24000 frames per second (fps)
"""
from config import get_config
cfg = get_config()
from splib.cute_dialog import start_dialog
import splib.toolbox as tbx
import os
import time
import subprocess
from collections import Counter

class G:
    pass

class dialog:
    recording = "hus1h"
    replace   = False

    layout = """
    title   Convert video file into wav format
    text    recording  recording id, example: 'hus1h'
    bool    replace    replace previously converted files
    """

def main():
    if not start_dialog(__file__, dialog):
        return

    G.stats = tbx.Statistics()

    base_path = str(cfg.recording).format(recd = dialog.recording)
    video_path = base_path + "/download"
    audio_path = base_path + "/source"

    print(video_path, audio_path)

    for fn in os.listdir(video_path):
        full = video_path + '/' + fn
        print(full)
        name, ext = os.path.splitext(fn)
        if not ext in ('.mkv', '.webm'):
            G.stats["ignored"] += 1
            continue
        ofile = audio_path + '/' + name + '.wav'
        if not dialog.replace and os.path.exists(ofile):
            G.stats["skipped"] += 1
            continue

        convert_to_wav(video_path, name, ext, audio_path)
        G.stats["converted"] += 1

    print(f"elapsed time: {G.stats.runtime()}")
    print(f"statistics:")
    print(G.stats.show())
    return


def convert_to_wav(ipath, name, ext, opath):
    ifile = ipath+'/'+name+ext
    ofile = opath+'/'+name+'.wav'
    cmd = ["ffmpeg", "-i",  ifile, "-ar", str(cfg.framerate), "-y", "-ac", "1", "-f", "wav", ofile]
    print('start subprocess', cmd)
    rc = subprocess.call(cmd)
    return rc

main()
