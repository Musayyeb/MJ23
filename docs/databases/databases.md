# The Project Databases

## Overview
The project uses PySQLite databases. 

The databases for different recordings are kept separate. These database files have the same name, but are stored in different recording-specific folders.

One database is reserved for project related data. This database file is stored in the root folder.

## DB Schema
There is one module **splib/db_schema.py**, which contains the specifications of all databases. These specifications include the database names, the tables and all key and field lists.

The db_schema.py is also an executable module. It can create the database files from the schema. Existing files will not be overwritten. To overwrite the files, they must be deleted manually.