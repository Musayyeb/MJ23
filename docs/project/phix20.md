Phonemics 2.0
=============

*Restart of the Phonemix project*

02.07.2022

Outline of the Project
----------------------
* Preparation steps
* Text Source: the Koran
* Audio Source
* Evaluate audio attributes
* Manual map training data
* Machine Learning: Complete Predictions
* Matrix Mapping: Complete letter mapping 
* Evaluate Phonem layouts and Synthesis
* Rythm and Melody - Introduce Prosody
* Project Structure
* Technical Infrastructure

* Smart mobile keyboard

## Preparation steps

There are way too many project environments living on Gitlab and local drives.
To reduce the chaos from the start, lets clean up existing environments.

Lets use at most 3 environments:
* The existing Speech-APS project
* This Phix20 project
* One archive project, where we collect snippets from previous ideas and codes.

### Cleanup existing projects

## -------------------------------------------

## Text Source: The Koran

The Koran text was processed in phase 1. 
We can use the transliteration for the Hafs reading, which we created before. 
There r other readings, like Warsh, QAAluun, and [about 10] others. 
Each of those readings has its own Koran text, with its own particular Alphabets and vowels and pronunciations. 
We will need to do different Transliterations for the different readings. 
We had however discussions about replacing / modifying the Hans Wehr transliteration. 
One idea, was to use the IPA alphabet to to represent the koran text. 
There r around 10 different readings of the Koran, and we need to go through those to see what we may encounter in different pronunciations. 
When we get a full list of those sounds, then we can move to IPA.

### Use of IPA for the Koran transliteration

This may turn into a little project of its own.
For now we do not use IPA

The koran text is adjusted according to the actual reading. The text therefore
is formatted to different blocks.
To make the enumeration of the blocks consistent, run the textt_input/enumerate_blocks.py

Then run text_input/count_blocks.py to have the short reference of the blocks per chapter.


## -------------------------------------------

## Audio Source

Currently we use the reading of Hussary. 

https://www.youtube.com/watch?v=iVrjdS-13Qw&list=PL7FQ8_TtkWWGIBbI_pRpjKbjgpLI7yYeg

The audio quality is limited, but at least it comes without echo, and with 
little background noise. Some other reciters have good voices, too, if we could
have access to material without echoes, that would be interesting. However, Hussary has many full different readings that r clear and without echo. we can use those for now maybe.

More information about the selected audio sources and about the processing steps
can be found in the docs/sound_input folder.

## -------------------------------------------

## Evaluate audio attributes

This part of the project shall define the "Features" of the audio data. 

The features of the audio are the basic data values needed for machine learning. 
The features are used for training and later for the predictions.
Essentially the features are just numbers. They may represent easyly identifyable attributes like loudness or frequency, but the larger part of the features are numbers derived mathematically from the audio data via some sophisticated algorithm (FFT, Formants, others).

In the previous approach to this we used about 25 features values, most of them being butterworth frequency bands. With 25 features, the Random Forest ML model gave us usable results. Using Keras with extra layers did not improve the results. My (H.) conclusion is, that the Keras model is just "underwhelmed" with 25 features.

The ML literature has proposals, how to go for audio classification. There are the LIBROSA package and the PyTorch-Audio package as starting points.

If we can prepare 100-200 features per audio event, then Keras will have a better time, crunching numbers and spitting out predictions.

Sound happens in time. To get anything audible, an audio file must produce air waves, which reach the ear. A .wav audio file may give 24000 or 48000 values (frames) per second, each of which represents one point on the audio wave line.

To represent one sine wave of 100 Hz, 240/480 frames are required, which take 
1/100 of a second (10 ms). The FFT - algorithm needs at least 2 complete waves,
to produce a signal for a certain frequency. To detect frequencies down to 80
or 90 hz, FFT usually requires audio frames for 25 or 30 ms.

30 ms is a lot of time in a speech signal. As speech attributes change rapidly,
an analysis window of 30 ms may already blur some audio attributes. There is
no such thing as precision in audio processing. 

Here FFT is just one example for different algorithms, which return attribute 
values for sound. To represent attributes of a continuous audio stream, we have
to decide for an interval, where we pick discrete attribute values. Values
between 5 and 10 ms are often used. Lets take 5.

Just a little bit of math:
10 hours of audio material gives 10 * 3600 * 200 gives 7.2 Million datapoints.
We collect 50 data points (numbers = 4 bytes) per data point, this already 
gives us 1.5 GB of data. We need to split this workload into reasonable chunks,
which can be processed individually (sequential / parallel)


## -------------------------------------------

## Manual Mapping of training data

We have developed a special application (qt_gui/manumap.py) with the QT toolkit. 
This app allows to load an audio block, load the appropriate transcribed 
text for that block, and then allow to assign a timestamp to each of the 
letters. The core functionality is the navigation along the wave display 
and the listening to a specific timespan. The manually mapped letters then go to a json file (workfiles/hus#h/manumap.json) where it can be collected, prepared and used for training. 

## -------------------------------------------

## Machine Learning: Complete Predictions

The process of Sound Classification happens in two steps:

1. Training of a model based on manually mapped training data
     

2. Predictions 

## -------------------------------------------

## Matrix Mapping: Complete letter mapping 

## -------------------------------------------

## Evaluate Phonem layouts and Synthesis

## -------------------------------------------

## Rythm and Melody - Introduce Prosody

## -------------------------------------------

## Project Structure
The layout of the project and data folders is mostly unchanged.

### config.py
This modules stores all global values and settings for the project.

### local_config.py
This module saves configuration data for the current enironment (which
means the user or the server environment). This file is not shared via
GIT.

### The data folder tree
This place is stored in the local config. It contains all the high-volume
and other data files, which are static (like the audio database) or are 
maintained by the user, but too big to share over GIT (like the SQL databases)

### The project folder
Contains all files, which are managed by PyCharm and shared via GIT push/pull.
There are exceptions to the sharing:
* sandbox is a folder for experimental code
* workfiles is a folder for program output

## -------------------------------------------

## Technical Infrastructure

There are a lot of tools and libraries, which where developed over time.
Most of the contain some good ideas and also some bad ideas. 

I (H.) feel the need to rebuild many of the tools, copying from the
old sources, what is worth to be copied.

### toolbox.py
Reuse the AttrDict object

### cute_dialog.py
The start_dialog window has to be made smarter, and it will be needed in almost 
all modules

## -------------------------------------------

## Smart mobile Keyboard

This is a project on its own. 

The best option would be a multiplatform development environment, which
produces the application for Android, IOS, and preferably for the browser.

Searching for possible platforms gives a few results

React Native
Ionic
QT

These (and other options) have to be verified as being fit for the 
development of a 'System' keyboard. Also some of the platforms may create
apps, which depend on a 'Server Backend' - This will exclude them.

From the Kotlin website, there is the information, that IOS native apps 
can only be created from the Mac/OS. 