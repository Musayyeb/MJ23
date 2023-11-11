Phonemics 2.0 - Audio input processing steps
============================================


23.07.2022

## Download and convert the files
Use the Firefox plug-in VideoDownloadHelper.

* Move the downloaded files from the dwhelper folder to the download folder of the respective recording
* Rename the video files with **rename_download.py** module
* If the file name does not contain the chapter number, rename it manually.
* Convert the videos into audio format (.wav) with the **convert_video_file.py** module. The output audio format is set to mono and 24000 frames per second.
## -------------------------------------------

## Identify block boundaries
The audio stream for each chapter need to be analyzed to identify the gaps between the blocks. This is done in the **find_blocks_in_source.py**. 

The program first has to convert the wav file into a loudness sequence with a 1ms resolution. Luckily this code is now super fast, thanks to the numpy library. There is no need to save the loudness sequence in a file, we can obtain it just in real-time. The function to do this is **get_amplitude_average()**, which lives in the **sound_input_lib.py** module

To identify the gaps, we must specify a minimum length for a gap (silence in the audio). Sometimes the reciter extends a plosive consonant (like q or t) quite long, so that it may appear like a gap. Depending on the specified gap length, the program may find false positives (a time of silence is taken for a gap) or false negatives (missing a real gap which is too short).

The min_gap specification must be calibrated for each recording, but is fixed for all chapters for a specific recording. Any choice will produce either false positives, false negatives or a mix of both. 

To identify errors in the gap assignemnt, we use the **check_block_boundaries.py**.

As it is much easier for the code to eliminate false positive gaps (just provide a "nogap" data table), it is probably better, to use a shorter min_gap parameter.

## -------------------------------------------

## Storage of Sound Attributes
For the huge quantities of data, like frequency, loudness and many other audio attributes, we need a way to store this data in the filesystem.

Numpy allows, to write data arrays to files, with exactly the same size, that is required to store the data in memory. We can write the loudness of a block, which is float, to a file at 4 bytes (float32) per interval (millisecond)

Example:

A wav file of 11 MB lasts about 4 minutes. The wav file is stored as int16 frames, which is 4 * 60 * 24000 * 2 bytes =~ 11 MB

To save the the loudness requires 4 * 60 * 1000 * 4 = 960 KB of filesystenm space.

    a = np.array([3.3, 4.4, 5.2, 7.9, 9.1], dtype='float32')
    with open('float.data', mode='wb') as fo:
        a.tofile(fo)
    
    with open('float.data', mode='rb') as fi:
        a = np.fromfile(fi, dtype='float32')

## Get the audio attributes
   All data attribute vectors come as a numpy array. 
   
Numpy can store any vector as a binary file in the file system. This is a very nice feature, because it allows to store data with a minimum usage of fileszstem space. Also it is quite fast and simple.

The audio attributes include things like amplitude, frequeny, zero-crossing-rates and more.

The machine learning engine will be trained and queried on the base of **5 millisecond interval datapoints**. All attributes are therefor stored in 5 ms interval vectors.

It is expected, that the 5ms interval is precise enough to accomplish all tasks of mapping / phonem cutting / synthesis. If not, we can always interpolate the vectors for smaller intervals.

