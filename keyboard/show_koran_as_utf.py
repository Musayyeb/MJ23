# python3
'''
    read the original koran text with all dos and diacritics
    show the sequence of utf codes
'''

from config import get_config
cfg = get_config()


koran_file = cfg.data / "koran_arabic_text.txt"


def main():
    diacsubs = {'50': '1', '51':'9', '52':'0', '4e':'3', '4f':'2',
                '514e':'6', '5150':'4', '514f':'5',
                }       # 50:i, 52:0, 4e:a, 4f:u, 51:shadda
    print(diacsubs)
    diacs = '50 51 52 4e 4f'.split()
    with open(koran_file, mode='r') as fi:
        data = fi.read(1500)

    words = data.split()

    for word in words:
        curr = []
        diac = []
        skip = 0
        for ndx, ch in enumerate(word):
            if skip:
                skip -= 1
                continue

            if ch in ('()0123456789'):
                continue

            hx = hex(ord(ch))
            pr = '\n' if hx == "0xa" else hx[-2:]

            if pr in diacs:
                p2 = hex(ord(word[ndx + 1]))[-2:] if len(word)-1 > ndx else ''
                if p2 == '51':  # means shaddha
                    diac.append(p2+pr)
                    skip = 1
                    continue
                diac.append(pr)
            else:
                curr.append(pr)

        curr_utf = '.'.join(curr)
        curr_arab = ''.join([chr(int('06' + utf, 16)) for utf in curr])
        print(curr_arab, ' - ', word)
        print(curr_utf, ' - ', ''.join([diacsubs[d] for d in diac]), end=' ')
        print()
    print(data)

main()