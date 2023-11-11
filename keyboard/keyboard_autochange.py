#python
"""
    Reference implementation of the keyboard autochange code
"""
from config import get_config
cfg = get_config()
import openpyxl as opx
from splib.cute_dialog import start_dialog

class dialog:
    tabname = "latin1"
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
    tab = read_xlttab(dialog.tabname)

    if dialog.showtab:
        for k, v in tab.items():
            print(f"{k:4}-> {v}")
        print('\n')

    process(dialog.intext, tab)

def process(inp, tab):
    out = []
    for ltr in inp:
        out.append(ltr)
        print(f"text: {''.join(out)}")
        chfl = True
        while chfl:
            chfl = False

            for strl in range(9,0, -1):
                if strl <= len(out):

                    check = ''.join(out[-strl:])
                    if check in tab:
                        print(f"matched {check} -> {tab[check]}")
                        out[-strl:] = list(tab[check][1])
                        print('          changed text:', ''.join(out))
                        chfl = True
                        break

    print(f"final output: {''.join(out)}")

    return


def read_xlttab(tabname):

    tabfile = cfg.ldata / 'detransliteration_4_keyboard.xlsx'
    wb = opx.load_workbook(filename=tabfile)  # read the excel (.xlsx) file
    rules_sheet = wb[tabname]

    tabdict = {}

    for rx, o_rule in enumerate(rules_sheet.iter_rows(max_col=2, values_only=True)):
        #  print(rx, o_rule)
        c1, c2 = o_rule[0], o_rule[1]
        if c1 is None or c2 is None or rx == 0:
            continue
        if c1 in tabdict:
            print(f'duplicate @{rx + 1} -->{c1} -{c2}   ?{tabdict[c1]}')
        tabdict[c1] = rx+1, c2

    return tabdict

main()