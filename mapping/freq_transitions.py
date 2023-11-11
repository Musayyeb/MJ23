

import splib.attrib_tools as att


def main():
    recd = "hus9h"
    chapno = 20
    blkno = 5

    vect_loader = att.VectLoader(recd, chapno, blkno)
    pyaa, rosa, freq = vect_loader.get_vectors()
    xdim = vect_loader.get_xdim()
    print(f"xdim {xdim}")
    print(f"len: {len(freq)}")
    ff = False  # freq.flag
    offst = 0
    for ndx, f in enumerate([int(x) for x in freq]):
        if ff:
            if f < 20:
                offst = ndx
                ff = False
        else:
            if f > 40:
                print(f"{offst:5d}:{ndx:5d} off  len={ndx-offst}" )
                ff = True

    print(f"{offst:5d}:{ndx:5d} off  len={ndx-offst}")

main()