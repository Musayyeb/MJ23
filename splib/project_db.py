# python3
"""
    Interface module for the new (chapter-aware) speech database
"""
from config import get_config
cfg = get_config()
from splib.db_schema import get_db_schema, show_dbs
import sqlite3
import os, os.path
from contextlib import contextmanager, ExitStack
import os, os.path


# print("sqlite version:", sqlite3.sqlite_version)


class G:
    db_names = []
    db_dict = {}
    recd = ''

def lib_init():
    # initialize this library
    # show_dbs()
    dbs = get_db_schema()

    G.db_names = dbs.dbnames()
    G.db_dict = {dbid : dbs.get_db(dbid) for dbid in dbs.dbnames()}
    print(G.db_names)

    # for i in G.db_dict.items():  print(i)


# =======================================================================
# =======================================================================

def db_connector(func, vdict=None):
    with DB_Manager() as dbman:
        if vdict is None:
            rc = func(dbman)
        else:
            rc = func(dbman, vdict)
    return rc


class DB_Manager(ExitStack):
    def __init__(self):
        super().__init__()
        self.connections = {}

    def connect(self, recd, dbid):
        #print(f"DB_Manager! dbnames:{G.db_names} dict:{G.db_dict}")
        dbkey = (recd, dbid)
        if dbkey in self.connections:
            return self.connections[dbkey]

        if not dbid in G.db_names:
            raise Exception(f"db {dbid} not defined")  # todo: Think about own exceptions

        db = ProjectDB(recd, dbid)
        conn = self.enter_context(db.connection())
        self.connections[dbkey] = conn
        return conn


class ProjectDB:
    def __init__(self, recd, dbid):
        # reording, database-id determine the database file name
        if not dbid in G.db_names:
            raise Exception(f"ProjectDB: dbid '{dbid}' not defined")

        self.dbnode = G.db_dict[dbid]  # DB Schema definitions for this database
        scope = self.dbnode.scope
        name = self.dbnode.dbfn + '.db'
        folder = cfg.data if scope == "root" else cfg.recording

        self.dbfn = str(folder / name).format(recd=recd)

        if not os.path.exists(self.dbfn):
            raise FileNotFoundError(f"missing db file: {self.dbfn}")

    @contextmanager
    def connection(self, autocommit=True):
        class Conn:
            def __init__(self, conn, db):
               self.conn = conn
               self.this_db = db

        print(f"connection")
        # this is a context manager for a db-connection
        # create a connection, which is closed automagically
        conn = sqlite3.connect(self.dbfn)


        co = Conn(conn, self)
        yield co  # this appears at the with statement

        if autocommit:
            co.conn.commit()
        co.conn.close()


def sql4insert(tbnode, fnames=None):
    # when fnames is None, take the fnames from the schema
    tbname = tbnode.tbname

    if not fnames:
        fnames = tbnode.get_fnames()
    qmarks = ', '.join('?' * len(fnames))
    fnames = ', '.join(n for n in fnames)

    sql = f""" INSERT INTO {tbname}
            ({fnames})
            VALUES({qmarks}) """

    return sql



lib_init()

if __name__ == '__main__':
    print("this is for import only")
