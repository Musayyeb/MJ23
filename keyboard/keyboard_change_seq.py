#python
"""
    Reference implementation of the keyboard autochange code
    This version does the change by processing the table top-down
"""
from config import get_config
cfg = get_config()
import openpyxl as opx
from splib.cute_dialog import start_dialog

xls_filename = "Latin1.xlsx"
solar = 'ث ص ض ن ت س ش ر ز د ذ ط ظ '
lunar = 'ج ح خ ه ع غ ف ق ك م ب ي و'

space = chr(0x61e)   # triple dots
subsol = chr(0x66d)  # asterisk
sublun = chr(0x61f)  # question mark

class dialog:
    tabname = "Sheet1"
    intext = 'example'
    showtab = False


    layout = """
    title  Experimental auto-complete keyboard
    text    tabname    translation table page in excel file
    text    intext     input text
    bool    showtab    print the table content
    """

def main():
    if not start_dialog(__file__, dialog):
        return

    #input_data = 'خَلَقَكُمْمِنْنَفْسٍوَاحِدَةٍوَخَلَقَمِنْهَا'
    tab = read_rules(dialog.tabname)

    if dialog.showtab:
        for k, v in tab.items():
            print(f"{k:4}-> {v}")
        print('\n')

    process(dialog.intext, tab)

def process(inp, tab):
    out = space   # initial value, string starts with space
    for ltr in inp:
        out += ltr
        print(f"text: {out}")

        for rx, l, fr, to in tab:
            flag, newout = match(out, fr, to)
            if flag:
                out = newout
    print(f"final output: {out}")

    return

def match(text, frpat, topat):
    pl = len(frpat)
    test = text[-pl:]
    tl = len(test)
    if frpat == test:
        print(f"'{test}' --> '{topat}'")
        text = text[:-tl] + topat
        return True, text

    p = frpat.find(subsol+subsol)
    if tl >= p+2 and p > -1:  #found two lunar symbols
        t1, t2 = test[p],test[p+1]
        if t1 == t2 and t1 in solar:
            frpat = frpat.replace(subsol, t1)
            if frpat == test:
                topat = topat.replace(subsol, t1)
                print(f"'{test}' --> '{topat}'")
                text = text[:-tl] + topat
                return True, text

    p = frpat.find(sublun+sublun)
    if tl >= p+2 and p > -1:  #found two lunar symbols
        t1, t2 = test[p],test[p+1]
        if t1 == t2 and t1 in lunar:
            frpat = frpat.replace(sublun, t1)
            if frpat == test:
                topat = topat.replace(sublun, t1)
                print(f"'{test}' --> '{topat}'")
                text = text[:-tl] + topat
                return True, text


    p = frpat.find(subsol)
    if tl > p and p > -1:  #found two lunar symbols
        t1 = test[p]
        if t1 in solar:
            frpat = frpat.replace(subsol, t1)
            if frpat == test:
                topat = topat.replace(subsol, t1)
                print(f"'{test}' --> '{topat}'")
                text = text[:-tl] + topat
                return True, text

    p = frpat.find(sublun)
    if tl > p and p > -1:  #found two lunar symbols
        t1 = test[p]
        if t1 in lunar:
            frpat = frpat.replace(sublun, t1)
            print(f" match? '{frpat}'  '{test}'")
            if frpat == test:
                topat = topat.replace(sublun, t1)
                print(f"'{test}' --> '{topat}'")
                text = text[:-tl] + topat
                return True, text

    return False, ''

def read_rules(tabname):

    tabfile = cfg.work / xls_filename
    wb = opx.load_workbook(filename=tabfile)  # read the excel (.xlsx) file
    rules_sheet = wb[tabname]

    repltab = []

    for rx, o_rule in enumerate(rules_sheet.iter_rows(max_col=2, values_only=True)):
        #  print(rx, o_rule)
        c1, c2 = o_rule[0], o_rule[1]
        if c1 is None or c2 is None or rx == 0:
            continue
        repltab.append((rx, len(c1), c1, c2))

    return repltab

main()