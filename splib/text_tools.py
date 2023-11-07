# python3
"""
    text_tools.py

    Collection of functions to process koran text data
"""
from config import get_config
cfg = get_config()
import splib.project_db as pdb


def get_block_text(recd, chapno, blkno, spec=None):
    db = pdb.ProjectDB(recd, 'recd')
    cbkey = f"{int(chapno):03d}_{int(blkno):03d}"
    sql = f"SELECT text FROM 'block_text' where cbkey=?"
    with db.connection() as co :
        csr = co.conn.execute(sql, (cbkey,))
        text = csr.fetchone()[0]
    if spec == "ml":
        text = mltext(text)
    return text



def mltext(text):
    # return a text, that is ready for ml mapping. The letter index (lndx) must be stable
    # we dont need dots and we don't need loooong vowels
    text = text.replace('.', '')
    for v in 'auiANWY':
        while True:
            l = len(text)
            text = text.replace(v*3, v*2)
            if len(text) == l:
                break
    return text
