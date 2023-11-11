# python3
""""""'''
    read the koran text (arabic) and count the words

'''
from config import get_config
cfg = get_config()
from collections import Counter
diacritics = (0x618, 0x619, 0x61a, 0x64b, 0x64c, 0x64d, 0x64e, 0x64f, 0x650, 0x651, 0x652, 0x633)

particle_tab = [
    [0x648, 0x64E, 0x0644, 0x650, 0x644],   # ولل
    [0x648, 0x64E, 0x0643, 0x64E],          # وك
    [0x648, 0x64E, 0x0628, 0x650],          # وب
    [0x648, 0x64E, 0x0644, 0x650],          # ول
    [0x648, 0x64E, 0x0621, 0x64E],          # وء
    [0x641, 0x64E, 0x0644, 0x650],          # فل
    [0x621, 0x64E, 0x0641, 0x64E],          # ءف
    [0x621, 0x64E, 0x0648, 0x64E],          # ءو
    [0x621, 0x64E, 0x0628, 0x650],          # ءل
    [0x621, 0x64E, 0x0644, 0x650],          # ءب
    [0x621, 0x64E, 0x0643, 0x64E],          # ءل
    [0x641, 0x64E, 0x0643, 0x64E],          # ءك
    [0x641, 0x64E, 0x0644, 0x650],          # فك
    [0x641, 0x64E, 0x0628, 0x650],          # فل
    [0x641, 0x64E, 0x0648, 0x64E],          # فب
    [0x644, 0x650, 0x0644],                 #لل
    [0x648, 0x64E],                         # و
    [0x641, 0x64E],                         # ف
    [0x643, 0x64E],                         # ك
    [0x628, 0x650],                         # ب
    [0x644, 0x650],                         # ل
    [0x621, 0x64E],                         # ء
    [0x627, 0x644],                         # ال

]

    # also remove last diacritic and tanween

def main():
    fn = str(cfg.data / 'koran_arabic_text.txt')
    with open(fn, mode='r') as fi:
        data = fi.read()

    # process "particles"
    particles = []
    # some characters with special meaning are stored as a part of the words
    # but should be separated by space
    for utf in particle_tab:
        particle = ''.join(chr(x) for x in utf)
        particles.append(particle)

    print(particles)

    prefixed, native = [], []


    words = data.split()
    for word in words:
        for part in particles:
            if word.startswith(part):
                prefixed.append(word)
                break
        else:
            native.append(word)




    ctnativ = Counter(native)
    ctprefx = Counter(prefixed)
    print("native", len(native), len(ctnativ), ctnativ)
    print("prefixed", len(prefixed), len(ctprefx),  ctprefx)

    split = nosplit = 0
    newwords = []
    for word in words:
        if word in ctnativ:
            newwords.append(word)
        else:
            # this is a prefixed
            for part in particles:
                if word.startswith(part):
                    l = len(part)
                    pref, stem = word[:l], word[l:]

                    if stem in ctnativ:
                        # only if the stem exists as native, we store prefix + stem
                        newwords.append(pref)
                        newwords.append(stem)
                        split += 1
                    else:
                        newwords.append(word)
                        nosplit += 1
                    break
            else:
                raise Exception(f"should start with a particle {word}")

    print(f"splitted {split}, not splitted {nosplit}")

    ctnewwords = Counter(newwords)
    print(f"new words {len(newwords)}, {len(ctnewwords)}, {ctnewwords}")
    return



    for code in diacritics:
        f = data.count(chr(code))
        print(hex(code), f)
        data = data.replace(chr(code), '')


    words = data.split()


    #print(words[:200])
    ct = Counter(words)
    print(ct)

    print(len(ct))

    one = words[200]
    print(one, [hex(ord(x)) for x in one])
    return


main()