# python3
"""
    The project runs in various environments (like your machine, my machine or the server).
    The configuration settings, which are specific to each environment are defined here

    This module is imported. It lives inside the code base, but must not be included into the GIT project.
"""
from pathlib import Path
import os

home = Path.home()
data = home / 'Data/ph20/'

recording = data / "{recd}"
loudness = data / "{recd}" / "loudness"
blocks   =  data / "{recd}" / "blocks"

audio_path = data / 'audio'
write_to_log = False