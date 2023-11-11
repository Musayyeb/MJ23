Phonemics 2.0 - Manual Mapping
============================================

*Data to feed the machine learning beast*

21.10.2022

## manumap.py - Tool for manual mapping
This gui application allows to easily map the text of the koran to positions in the sound files (blocks).

The tool depends on the stability of the left block boundary. At some point in the project the block boundaries were changed, which leads to a a situation, where previously mapped data get invalid. This must be corrected.

The mappings are stored in a database table. The table contains the letter (label), the letter index (position of the letter in the text) and the millisecond position of the letter. The text is used in the **ml** format, that is without dots and without repeated (more than 2) vowels.

