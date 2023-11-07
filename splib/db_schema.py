
db_schema = """#  database schema

#===========================================================================

# database for basic / central / general tables 


db_id:   proj / project / root

#########   lettermap

table:  mapping_done
                        # after mapping a block, write one record here, 
                        # if the record is here, skip the mapping
    uniq:  recd chapno blkno 
    recd    text  nn  # individual letter
    chapno  int   nn  # recording
    blkno   int   nn


table:  lettermap
    uniq:  ltr recd cbkey lndx
    ltr    text  nn  # individual letter
    recd   text  nn  # recording
    cbkey  text  nn
    lndx   int   nn  # letter index of the first letter 
    rept   int   nn  # repetition: add this to the current lndx to get the next lndx
    ratg   float     # some value, which represents the 'quality' of the letter, higher is better
    lpos   int   nn  # left position (ms)
    rpos   int   nn  # right position (ms)
    tlen   int       # diff betw. lpos and rpos
    lgap   int   nn  # gap to the previous letter (negative if overlap)
    rgap   int   nn  # gap to the next letter (negative if overlap)
    lampl  int       # amplitude (loudness) at the left boundary
    rampl  int       # amplitude (loudness) at the right boundary
    lfreq  int       # freq at the left boundary
    rfreq  int       # freq at the right boundary
    melody text      # text represents a number of intermediate frequencies


#table:  phonem_1
#    # cut single letter phonems according to lgap/rgap
#    uniq:  ltr recd cbkey lndx
#    ltr    text  nn  # individual letter
#    recd   text  nn  # recording
#    cbkey  text  nn
#    lndx   int   nn  # letter index of the first letter 
#    lcut   int       # left cut position
#    rcut   int       # right cut position
#    lampl  int       # amplitude (loudness) at the left boundary
#    mampl  int       # amplitude (loudness) in the middle
#    rampl  int       # amplitude (loudness) at the right boundary
#    lfreq  int       # freq at the left boundary
#    mfreq  int       # freq in the middle
#    rfreq  int       # freq at the right boundary

    
#table:  phonem_2
## cut lingle letter phonems at the freq/no-freq boundaries
#    uniq:  ltr recd cbkey lndx
#    ltr    text  nn  # individual letter
#    recd   text  nn  # recording
#    cbkey  text  nn
#    lndx   int   nn  # letter index of the first letter 
#    lcut   int       # left cut position
#    rcut   int       # right cut position
#    lampl  int       # amplitude (loudness) at the left boundary
#    mampl  int       # amplitude (loudness) in the middle
#    rampl  int       # amplitude (loudness) at the right boundary
#    lfreq  int       # freq at the left boundary
#    mfreq  int       # freq in the middle
#    rfreq  int       # freq at the right boundary

    
    
#table:  diphone
## cut single letter phonems at the freq/no-freq boundaries
#    uniq:  name recd cbkey lndx
#    name   text  nn  # single consonant or diphon
#    recd   text  nn  # recording
#    cbkey  text  nn
#    lndx   int   nn  # letter index of the first letter 
#    lcut   int       # left cut position
#    rcut   int       # right cut position
#    lampl  int       # amplitude (loudness) at the left boundary
#    rampl  int       # amplitude (loudness) at the right boundary
#    lfreq  int       # freq at the left boundary
#    rfreq  int       # freq at the right boundary

#table: cvcphone
##    uniq: name recd cbkey lndx
#    name   text nn  # 2 to 4 letter name
#    patt   text nn  # pattern string of the phonem
#    ratg   float    
#    recd   text nn  # recording
#    cbkey  text nn
#    lndx   int  nn  # letter index of the vowel / main consonat
#    vrept  int      # repeat of the vowel
#    crept  int      # repeat of the consonant after the vowel
#    lcut   int  nn  # left cut position
#    rcut   int  nn  # right cut position
#    tlen   int      # total length
#    lampl  int      # amplitude (loudness) at the left boundary
#    rampl  int      # amplitude (loudness) at the right boundary
#    vampl  int      # amplitude of the vowel (average? amplitude)
#    lfreq  int      # freq at the left boundary
#    rfreq  int      # freq at the right boundary
#    lvfreq int      # boundary frequencies of the vowel
#    rvfreq int      #   "

#table: dummy
#    cbkey  text  nn
#    dummy int



#table: yuppie
#    cbkey  text  nn
#    other  float


#===========================================================================


# database for recording specific tables 

db_id:   recd / recording

#########   textmap

table:  textmap
    uniq:  name cbkey lndx
    name   text  nn  # syllable or individual letter
    cbkey  text  nn
    lndx   int   nn  # letter index of the first letter 
    lpos   int   nn  # left position (ms)
    rpos   int   nn  # right position (ms)
    lampl  int       # amplitude (loudness) at the left boundary
    rampl  int       # amplitude (loudness) at the right boundary
    lfreq  int       # freq at the left boundary
    rfreq  int       # freq at the right boundary

#########   block text - the koran text

table:   block_text
    uniq: cbkey
    cbkey  text   nn  # default key format ccc_bbb
    text   text     # string with leading and trailing dots, no spaces 


#===========================================================================

# database for machine-learning related data
# recording specific

db_id:   ml01 / mldata_01

#########   manually mapped label to sound

table:   manumap
    cbkey    text   nn   # default key format ccc_bbb
    msoffs   int    nn   # ms offset, center of the sound 
    label    text   nn   # letter or symbol
    lndx     int    nn   # position in the text (1st = 0)


table:   train_data
    uniq:    cbkey msoffs
    cbkey      text   nn
    msoffs     int    nn   # always a multiple of 5
    label      text   nn   # the letter
    # here come the attribute values
    my_ampl    float       # amplitude calculated from own averaging algorithm
    rosa_ampl  float
    pars_ampl  float
    pars_freq  float
    pars_fmnt_1  float       # formants
    pars_fmnt_2  float
    pars_fmnt_3  float
    pars_fmnt_4  float
    pars_fmnt_5  float
    pars_fmnt_6  float
    pars_fmnt_7  float
    pyaa_zcr   float
    pyaa_enrg  float       # energy 
    pyaa_enrg_entr  float  # energy entropy
    pyaa_spec_cent  float  # spectral centroid
    pyaa_spec_sprd  float  # spectral spread
    pyaa_spec_entr  float  # spectral entropy
    pyaa_spec_flux  float  # spectral flux
    pyaa_spec_rlof  float  # spectral roll-off
    pyaa_mfcc_1   float    # mel frequency
    pyaa_mfcc_2   float
    pyaa_mfcc_3   float
    pyaa_mfcc_4   float
    pyaa_mfcc_5   float
    pyaa_mfcc_6   float
    pyaa_mfcc_7   float
    pyaa_mfcc_8   float
    pyaa_mfcc_9   float
    pyaa_mfcc_10  float
    pyaa_mfcc_11  float
    pyaa_mfcc_12  float
    pyaa_mfcc_13  float
    pyaa_chrm_1   float     # chroma
    pyaa_chrm_2   float
    pyaa_chrm_3   float
    pyaa_chrm_4   float
    pyaa_chrm_5   float
    pyaa_chrm_6   float
    pyaa_chrm_7   float
    pyaa_chrm_8   float
    pyaa_chrm_9   float
    pyaa_chrm_10  float
    pyaa_chrm_11  float
    pyaa_chrm_12  float
    pyaa_chrm_std float
#===========================================================================

db_id:   ml02 / mldata_02

#########   automap  -  feed back mappings from successful predictions

table:   automap
    cbkey    text   nn   # default key format ccc_bbb
    msoffs   int    nn   # ms offset, center of the sound 
    label    text   nn   # letter or symbol
    lndx     int    nn   # position in the text (1st = 0)


table:   train_data
    uniq:    cbkey msoffs
    cbkey      text   nn
    msoffs     int    nn   # always a multiple of 5
    label      text   nn   # the letter
    # here come the attribute values
    my_ampl    float       # amplitude calculated from own averaging algorithm
    rosa_ampl  float
    pars_ampl  float
    pars_freq  float
    pars_fmnt_1  float       # formants
    pars_fmnt_2  float
    pars_fmnt_3  float
    pars_fmnt_4  float
    pars_fmnt_5  float
    pars_fmnt_6  float
    pars_fmnt_7  float
    pyaa_zcr   float
    pyaa_enrg  float       # energy 
    pyaa_enrg_entr  float  # energy entropy
    pyaa_spec_cent  float  # spectral centroid
    pyaa_spec_sprd  float  # spectral spread
    pyaa_spec_entr  float  # spectral entropy
    pyaa_spec_flux  float  # spectral flux
    pyaa_spec_rlof  float  # spectral roll-off
    pyaa_mfcc_1   float    # mel frequency
    pyaa_mfcc_2   float
    pyaa_mfcc_3   float
    pyaa_mfcc_4   float
    pyaa_mfcc_5   float
    pyaa_mfcc_6   float
    pyaa_mfcc_7   float
    pyaa_mfcc_8   float
    pyaa_mfcc_9   float
    pyaa_mfcc_10  float
    pyaa_mfcc_11  float
    pyaa_mfcc_12  float
    pyaa_mfcc_13  float
    pyaa_chrm_1   float     # chroma
    pyaa_chrm_2   float
    pyaa_chrm_3   float
    pyaa_chrm_4   float
    pyaa_chrm_5   float
    pyaa_chrm_6   float
    pyaa_chrm_7   float
    pyaa_chrm_8   float
    pyaa_chrm_9   float
    pyaa_chrm_10  float
    pyaa_chrm_11  float
    pyaa_chrm_12  float
    pyaa_chrm_std float
#===========================================================================


# database for test

#        dbid    dbfn    scope (root or recd)

db_id:   test / dbtest / root   # dbid/name/scope

table:   tabone
    xkey      text    nn
    number    int     nn
    amount    float
    joke      text
    dict      blob



#===========================================================================
"""


from dataclasses import dataclass, fields
from splib.toolbox import AD

class SchemaError(Exception):
    pass


def get_db_schema():
    dbs = Schema()
    curr_db = None
    curr_tb = None
    env = None

    for line, toks in get_lines():
        keywd = toks[0]
        if   keywd == 'db_id:':
            dbid, dbfn, scope = parse_dbid(toks)
            curr_db = DBase(dbid, dbfn, scope)
            dbs.add_db(curr_db)
        elif keywd == 'table:':
            tbname = toks[1]
            curr_tb = DbTable(tbname)
            curr_db.add_table(curr_tb)
        elif keywd == "uniq:":
            curr_tb.uniq = toks[1:]
        else:
            # all fields have a type, keys are also fields
            fname, ftype = toks[:2]
            fattr = '  '.join(toks[2:]) if len(toks) > 2 else ''
            ftype = check_type(ftype)
            fattr = check_attr(fattr)
            curr_tb.add_field(DbField(fname, ftype, fattr))

    return dbs

def check_type(t):
    t = t.upper()
    t = t.replace('FLOAT', 'REAL')
    return t

def check_attr(a):
    a = a.replace('key', 'PRIMARY KEY')
    a = a.replace('nn', 'NOT NULL')
    return a

def parse_dbid(toks):
    dbid = toks[1]
    dbfn = toks[3]  if len(toks) >= 4 else  dbid
    scope = toks[5] if len(toks) >= 6 else 'recd'
    if not scope in ('recd', 'root'):
        raise SchemaError(f"bad scope for {dbid}: '{scope}'")
    return dbid, dbfn, scope

def get_lines():
    for line in db_schema.splitlines():
        line = line.split('#')[0].rstrip()
        if line == '':
            continue
        toks = line.split()
        yield line, toks

def show_dbs():
    dbs = get_db_schema()
    for db in dbs.dblist:
        print("   ",db)
        for tb in db.tblist:
            print("      ", tb)
            for f in tb.flist:
                print("         ", f)

    return

@dataclass
class Schema:
    # Definitions of all database

    def __post_init__(self):
        self.dblist = []   # list of databse nodes

    def add_db(self, dbnode):
        self.dblist.append(dbnode)

    def dbnames(self):
        return [dbnode.dbid for dbnode in self.dblist]

    def get_db(self, dbid):
        for dbnode in self.dblist:
            if dbnode.dbid == dbid:
                return dbnode
        raise SchemaError(f"unknow database id: '{dbid}'")

@dataclass
class DBase:
    # definition of a database
    dbid: str
    dbfn: str
    scope: str

    def __post_init__(self):
        self.tblist = []  # list of table nodes

    def add_table(self, tbnode):
        self.tblist.append(tbnode)

    def tbnames(self):
        return [tbnode.tbname for tbnode in self.tblist]

    def get_table(self, tbname):
        for tbnode in self.tblist:
            if tbnode.tbname == tbname:
                return tbnode
        raise SchemaError(f"wrong table name: '{tbname}'")

@dataclass
class DbTable:
    tbname: str

    def __post_init__(self):
        self.flist = []   #  list of field nodes

    def add_field(self, fnode):
        self.flist.append(fnode)

    def get_fnames(self):
        return [fnode.fname for fnode in self.flist]

    def get_fields(self):
        return self.flist

    def get_field(self, fname):
        fn = [f for f in self.flist if f.fname == fname]
        if fn:
            return fn[0]
        raise SchemaError(f"wrong field name: '{fname}'")


@dataclass
class DbField:
    fname: str
    ftype: str
    fattr: str = ''

# ===========================================================================

from config import get_config
cfg = get_config()  # default config
from splib.cute_dialog import start_dialog
import sqlite3


class dialog:
    recording = "hus1h"

    layout = """
    title   Create databases and all tables
    text    recording  recording id, example: 'hus1h'
    label   Select the databases:
"""

def main():
    from splib import project_db
    import os, os.path
    import logging as lg


    dbs = get_db_schema()
    flags = '\n'.join([f'    bool  {n}  {n}' for n in dbs.dbnames()])
    dialog.layout = dialog.layout + flags

    # print(dialog.layout)
    if not start_dialog(__file__, dialog):
        return

    # This code creates a new database if it does not yet exist.
    # This code creates new tables from the db_schema module
    # It does not replace existing tables, so it is safe to run at any time.
    # To replace tables, delete them with the SqLite Browser tool
    # then run this code (again)

    # If a table (structure) has changed, we have to write code for exporting,
    # replacing the table and importing the data again.

    recd = dialog.recording

    for dbnode in dbs.dblist:
        dbname = dbnode.dbid
        flag = getattr(dialog, dbname)
        if flag:  # database name was selected in the dialog
            dbfn = dbnode.dbfn
            scope = dbnode.scope
            fn = dbfn + '.db'
            path = cfg.data if scope == 'root' else cfg.recording
            full = str(path / fn).format(recd=recd)

            # if os.path.exists(full):
            #    print(f"database {dbname} ({full}) already exixst")
            #    continue

            print(f"\nProcess {dbname} database for {full}\n")

            create_database_tables(dbnode, full)
    return


def create_database_tables(dbnode, dbfn):
    with sqlite3.connect(dbfn) as conn:
        # print(f"connection: {conn}")

        # find the existing tables
        print("showing existing tables:\n")
        tbnames = []
        csr = conn.execute('SELECT name, sql FROM sqlite_schema where type="table" ')
        for name, sql in csr.fetchall():
            tbnames.append(name)
            print("existing sql:\n", sql)

        print("\ncheck db_schema definitions:\n")
        for tbnode in dbnode.tblist:
            name = tbnode.tbname
            if name in tbnames:
                print(f"table '{name}' already exists")
                continue

            # create the sql for the missing table
            sql_text = [f"CREATE TABLE '{name}' "]
            sql_text.append('   (')
            sql_text.append("  id  INTEGER   PRIMARY KEY , ")
            fields = [f"  {f.fname}    {f.ftype}   {f.fattr} " for f in tbnode.flist]
            sql_text.append(", \n".join(fields))
            if hasattr(tbnode, "uniq"):
                sql_text.append(", UNIQUE (" + ", ".join(tbnode.uniq) +")")
            sql_text.append(")")
            sql = "\n".join(sql_text)
            print("new table sql:\n", sql)
            conn.execute(sql)

    return


@dataclass
class LMAP:
    id     : int
    ltr    : str  # individual letter
    recd   : str  # recording
    cbkey  : str  # {chapno:03d}_{blkno:03d}
    lndx   : int  # letter index of the first letter
    rept   : int  # repetition: add this to the current lndx to get the next lndx
    ratg   : float  # some value, which represents the 'quality' of the letter, higher is better
    lpos   : int  # left position (ms)
    rpos   : int  # right position (ms)
    lgap   : int  # gap to the previous letter (negative if overlap)
    rgap   : int  # gap to the next letter (negative if overlap)
    lampl  : int  # amplitude (loudness) at the left boundary
    rampl  : int  # amplitude (loudness) at the right boundary
    lfreq  : int  # freq at the left boundary
    rfreq  : int  # freq at the right boundary
    melody : str  # text represents a number of intermediate frequencies

@dataclass
class PH1:
    id     : int
    ltr    : str  # individual letter
    recd   : str  # recording
    cbkey  : str  # {chapno:03d}_{blkno:03d}
    lndx   : int  # letter index of the first letter
    lcut   : int  # repetition: add this to the current lndx to get the next lndx
    rcut   : int  # some value, which represents the 'quality' of the letter, higher is better
    lampl  : int  # amplitude (loudness) at the left boundary
    mampl  : int  # amplitude (loudness) at the left boundary
    rampl  : int  # amplitude (loudness) at the right boundary
    lfreq  : int  # freq at the left boundary
    mfreq  : int  # freq at the left boundary
    rfreq  : int  # freq at the right boundary

@dataclass
class DIPH:
# cut lingle letter phonems at the freq/no-freq boundaries
    id    : int
    name  : str  # single consonant or diphon
    recd  : str  # recording
    cbkey : str
    lndx  : int  # letter index of the first letter
    lcut  : int  # left cut position
    rcut  : int  # right cut position
    lampl : int  # amplitude (loudness) at the left boundary
    rampl : int  # amplitude (loudness) at the right boundary
    lfreq : int  # freq at the left boundary
    rfreq : int  # freq at the right boundary


@dataclass
class CVC:
# cut lingle letter phonems at the freq/no-freq boundaries
    id    : int
    name  : str  # 2 to 4 letter name
    patt  : str  # pattern string of the phonem
    ratg  : float
    recd  : str  # recording
    cbkey : str
    lndx  : int   # letter index of the vowel / main consonat
    vrept  : int # repeat of the vowel
    crept  : int # repeat of the consonant after the vowel
    lcut  : int  # left cut position
    rcut  : int  # right cut position
    tlen  : int  # total length
    lampl : int  # amplitude (loudness) at the left boundary
    rampl : int  # amplitude (loudness) at the right boundary
    vampl : int  # amplitude of the vowel (average? amplitude)
    lfreq : int  # freq at the left boundary
    rfreq : int  # freq at the right boundary
    lvfreq : int  # boundary frequencies of the vowel
    rvfreq : int  #   "

    def str(self):
        s = self
        return f"CVC: [{s.name}]  ldnx:{s.lndx}  vrpt:{s.vrept},crpt:{s.crept}   {s.recd},{s.cbkey} pos:{s.lcut, s.rcut}({s.tlen})   " \
               f"rtg:{s.ratg:5.2f}   vowfreq:{s.lvfreq, s.rvfreq} vampl:{int(s.vampl)}   tol:{s.tol}"

def fill_dataclass(class_name, arg_dict, defaults):
    # get all the fields from the dataclass definition
    # extract all key:value pairs from the defaults and the argdict
    # use the extracted dictionary to initialize the dataclass object
    arg_set = arg_dict.keys()
    field_set = {f.name for f in fields(class_name) if f.init}
    filtered_arg_dict = {k : v for k, v in arg_dict.items() if k in field_set}
    for k in field_set:
        if k not in arg_set:
            filtered_arg_dict[k] = defaults[k]
    return class_name(**filtered_arg_dict)


if __name__ == '__main__':
    dbs = get_db_schema()
    print(dbs)

    main()

