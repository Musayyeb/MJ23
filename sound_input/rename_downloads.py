# python3
"""
    Bulk rename of files in a download folder
    Some koran download files can be renamed automatically
    to the correct form of "chap_{03d}.ext. It can be done,
    if the name (mostly arabic) contains the chapter number
    in latin digits.
    The video files are located in the "download" folder
    of the selected recording

    This code can be safely repeated
"""
from config import get_config
cfg = get_config()
from splib.cute_dialog import start_dialog
import os

class dialog:
    recording = "hus1h"

    layout = """
    title   Rename video files into the required format chap_nnn.ext
    text    recording  recording id, example: 'hus1h'
    """

def main():
    if not start_dialog(__file__, dialog):
        return

    base_path = str(cfg.recording).format(recd = dialog.recording)
    video_path = os.path.join(base_path, "download")

    print("process video_path:", video_path)

    for fn in os.listdir(video_path):
        oname, ext = os.path.splitext(fn)

        name = get_chapter_name(oname)

        ofn = video_path + '/' + oname + ext
        nfn = video_path + '/' + name + ext
        if ofn == nfn:
            continue # same name, no action
        print(f"rename file to {name}")
        os.rename(ofn, nfn)

    return

def get_chapter_name(oname):
    # assume, that the name contains the chapter number in latin digits
    # and no other digits appear in the name
    chapno = "".join([x for x in oname if x in "0123456789"])
    if len(chapno) != 3:
        print(f"name does not match required format: {oname}")
        return oname
    return f"chap_{chapno}"

main()