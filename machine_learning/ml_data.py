#python3
'''
    prepare machine learning data
    here we have to specify, what part of the available data
    actually goes into the training

    We will have to try various different setting to dettermin which combination
    gives us the best result.
    We also look at the run time and at memory constraints.
    The runtime and memory requirements at the same time depend on the
    configuration of the model.
'''
from config import get_config
cfg = get_config()
import splib.project_db as pdb
import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
import numpy as np
import splib.attrib_tools as att
from splib.toolbox import AttrDict as AD
from collections import Counter
import random


class G:
    recd = ''
    short_attribs = ''  # final list of attributes

def get_pred_data(recd, chapno, blkno, spec):
    # get the data to predict the labels of a piece of audio attributes

    # load a block from the attribute numpy files
    all_np_vect = att.AllVectors(recd, chapno, blkno)
    vectors = all_np_vect.get_vectors()
    names = all_np_vect.get_names()

    attribs = attr_list(spec['attr_sel'])
    span = spec['span']
    #print("pred attrib names:", len(attribs))
    # prepare it for prediction for a model

    # select the vectors by order of the names in attribs
    sel_vect = [vectors[names.index(n)] for n in attribs]

    minlen = min([len(v) for v in sel_vect])
    sel_vect = [x[:minlen] for x in sel_vect]  # make sure all vectors have the same (smallest) length

    #print("sel_vector:", len(sel_vect), len(sel_vect[0]))
    #print(f"span: {span}")

    if span > 0:

        # there are two ways to stack the extended data slices, number 2 matches the training data
        np_vect = extend_attr_vect_2(sel_vect, span)
    else:
        np_vect = np.swapaxes(sel_vect, 0, 1)  # np.array(sel_vect)

    #print("np_vect", np_vect.shape)
    return np_vect


def extend_attr_vect_2(imat, span):
    # return the sequence of each atttribute separate for each span
    iter = int(span/5)  # span 5 or 10 gives 1 or 2 iterations
    mat = []  # output matrix


    slice_len = len(imat)
    #print(f"slice_len: {slice_len}")

    null_sl = [0]*slice_len  # extra empty slices at the beginning and the end

    sltab = []  # slice table contains 3 of 5 slices

    seq = (zip(*imat))  # a generator expression: return input matrix as sequence of slices

    for _ in range(iter):
        sltab.append(null_sl)  # zero slices

    for _ in range(iter):
        sl = next(seq)
        sltab.append(sl)      # data slices

    for sl in seq:            # now the main loop is ready
        sltab.append(sl)      # add one more data slice
        flatlist = [item for sublist in sltab for item in sublist]  # flatten nested list
        mat.append(flatlist)
        sltab.pop(0)          # remove the oldest slice

    for _ in range(iter):     # add the final slices
        sltab.append(null_sl)
        flatlist = [item for sublist in sltab for item in sublist]
        mat.append(flatlist)
        sltab.pop(0)

    return np.array(mat)

def extend_attr_vect_1(mat, span):
    # return spanned values for each attribute in sequence
    exp_vect = []

    l1 = l2 = r1 = r2 = []

    for vect in mat:
        # single vector with all values for a given attribute
        if span >= 5:
            l1 = np.roll(vect, 1)
            l1[0] = 0.0
            r1 = np.roll(vect, -1)
            r1[-1] = 0.0

        if span == 10:
            l2 = np.roll(l1, 1)
            l2[0] = 0.0
            r2 = np.roll(r1, -1)
            r2[-1] = 0.0

        print(f"span vectors: span={span}, {len(l2), len(l1), len(vect), len(r1), len(r2)}")

        if span == 10:    exp_vect.append(l2)
        if span >=  5:    exp_vect.append(l1)
        exp_vect.append(vect)
        if span >=  5:    exp_vect.append(r1)
        if span == 10:    exp_vect.append(r2)

    np_vect = np.array(exp_vect)
    np_vect = np.swapaxes(np_vect, 0, 1)

    return np_vect


def get_training_data(spec):
    # recd_s is a list of recording names (hus1h, hus9h) which may be empty or contain 1 or 2
    # attr_sel is a string of 5 letters to select attribute groups
    # span is a number (0, 5, 10) which specifies if the +/- 5/10 ms positions are incuded

    rc = pdb.db_connector(db_worker, spec)
    return rc

def db_worker(dbman, vdict):
    vspan = int(vdict['span'])
    span = 4 if vspan == 10 else 3 if vspan == 5 else 2
    recd_s = vdict['recd']
    reduced = vdict['reduced']

    toks = recd_s.split()  # get one or two names
    print('toks:', toks)
    if len(toks) == 1 and toks[0] == 'none':
        recds = []
    else:
        recds = toks
    print('recds:',recds)



    attribs = ['label', 'cbkey', 'msoffs']
    attribs.extend( attr_list(vdict['attr_sel']) )
    print("train attrib names:", len(attribs))



    df = get_dataframe(dbman, 'ml01', 'hus9h', 'manumap', attribs, span, reduced)
    for recd in recds:
        df_auto =  get_dataframe(dbman, 'ml02', recd, 'automap', attribs, span, reduced)
        df = df.append(df_auto)

    target_column = ['label']
    predictors = G.short_attribs[3:]
    print("final attrib names:", len(predictors))

    X = df[predictors].values
    R = df[['letter', 'cbkey', 'msoffs']].values
    y = df[target_column].values

    y = [t[0] for t in y]
    print(f"training data {X.shape}")
    return X, R, y

def get_dataframe(dbman, dbname, recd, tbname, attribs, span, reduced):
    # do the database processing
    dbref = dbman.connect(recd, dbname)
    conn = dbref.conn
    attr_str = ', '.join(attribs)
    sql = f"SELECT {attr_str} from 'train_data' WHERE LENGTH(label) < {span} ORDER BY cbkey, msoffs"
    # sql = f"""SELECT {attr_str} from 'train_data' WHERE label in ('a','f','l')
    #             ORDER BY id"""
    csr = conn.execute(sql)  # , ('?', ))
    lc = att.LabelCategories()  # static mapping of labels and categorical numbers

    ltrs = Counter()
    random.seed("fix")

    # print("basic attrib names", attribs)
    ext_attribs = attribs[:]
    for ltr in "abcd":
        ext_attribs.extend([n + ltr for n in attribs[3:]])

    # print("extended attrib names", ext_attribs)

    def join_rows(csr):
        # the training data returns (up to) 5 records for the letter attributes
        # the training data for that one letter is then extended into a long list
        joined = []
        prev_label = ''
        prev_cbkey = ''
        prev_pos = 0

        for row in csr:
            label, cbkey, msoffs = row[:3]

            # reduce test data
            ##if cbkey[:3] != "002":
            ##    continue

            if msoffs == prev_pos + 5 and cbkey == prev_cbkey:
                # then i am in the same mapping
                # add the current attribs to the collection
                joined.extend(row[3:])
                # print(f"joining spanned data: {len(joined)}")

            else:
                if joined:  # only if there was a previous element
                    # yield the previous mapping
                    yield [prev_label, prev_cbkey, prev_pos] + joined

                # then create a new mapping from the current row
                joined = [] + list(row[3:])  # append the current row to a new collection
                prev_label = label[0]
                prev_cbkey = cbkey
            prev_pos = msoffs

        # yield the last collection
        yield [prev_label, prev_cbkey, prev_pos] + joined

    def generate(csr):
        joiner = join_rows(csr)
        row_count = 0

        # there are only 42 attribut names in the database
        # here we include additional names for the +-5/+-10 ms positions
        # is has to be extended accoring to the actual length opf the data
        for row in joiner:
            # shorten the maximum list of attribute names to the required length
            short_attribs = ext_attribs[:len(row)]  # assume the length of row is always the same
            # print("short attribs:", short_attribs)
            label = row[0]
            # print(row[:3])

            if reduced:
                if label[0] in 'ḥḫsšḍfhṣġ' and random.random() > 0.2: continue
                if label[0] in 'qkṭǧbdt' and random.random() > 0.5: continue


            ltrs[row[0]] += 1
            # row = [0 if r is None else r for r in row]  # avoid NaN conditions

            rowdict = {l: v for l, v in zip(short_attribs, row)}
            # print("row dict:", len(rowdict), len(short_attribs), len(row))
            # print("row dict:", list(rowdict.keys()))
            rowdict['letter'] = rowdict['label']  # this is the original letter

            rowdict['label'] = lc.categ[rowdict['label']]  # convert labels to numbers

            row_count += 1
            if row_count > 50000:
                print("****** break at 50000 ... why?")
                break

            yield rowdict
        G.short_attribs = short_attribs
        # print("reduced attrib names:", short_attribs)
        # end of generator

    df = pd.DataFrame(generate(csr))

    return df


def attr_list(attr_sel):
    at_fmts = [f"pars_fmnt_{n + 1}" for n in range(7)]
    at_base = ['pyaa_zcr', 'pyaa_enrg', 'pars_freq', 'pars_ampl', 'rosa_ampl']
    at_pyaa = ['pyaa_' + s for s in '''enrg_entr spec_cent spec_sprd 
                                       spec_entr spec_flux spec_rlof'''.split()]
    at_chrom = [f'pyaa_chrm_{n}' for n in range(1, 13)] + ['pyaa_chrm_std']
    at_mfcc = [f"pyaa_mfcc_{n + 1}" for n in range(13)]

    attribs = []
    if 'b' in attr_sel:  attribs.extend(at_base)
    if 'c' in attr_sel:  attribs.extend(at_chrom)
    if 'f' in attr_sel:  attribs.extend(at_fmts)
    if 'm' in attr_sel:  attribs.extend(at_mfcc)
    if 'p' in attr_sel:  attribs.extend(at_pyaa)

    return attribs


def data_r_split(x_data, y_labels, r_data, test_size):
    # split dataset into training and test part
    # also keep track of a reference vector, which links
    # the test data to the database keys
    x_train, x_test, y_train, y_test, r_test = [], [], [], [], []

    for dx, dy, dr in zip(x_data, y_labels, r_data):
        if random.random() < test_size:
            x_test.append(dx)
            y_test.append(dy)
            r_test.append(dr)
        else:
            x_train.append(dx)
            y_train.append(dy)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, x_test, y_train, y_test, r_test


def data_split(x_data, y_labels, test_size):
    # split dataset into training and test part
    x_train, x_test, y_train, y_test = [], [], [], []

    for dx, dy in zip(x_data, y_labels):
        if random.random() < test_size:
            x_test.append(dx)
            y_test.append(dy)
        else:
            x_train.append(dx)
            y_train.append(dy)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, x_test, y_train, y_test


def test():
    spec = AD(attr_sel = 'b', span = 10, recd="none", reduced=True)
    X, R, y = get_training_data(spec)

def test_pred():
    spec = AD(attr_sel = 'bcfmp', span = 0)
    X = get_pred_data("hus9h", 2, 99, spec)

if __name__ == "__main__":
    test()