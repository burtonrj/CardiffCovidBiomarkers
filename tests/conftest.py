import sqlite3 as sql
import pytest


@pytest.fixture(scope='session', autouse=True)
def con():
    # Open database for first test
    con = sql.connect("/home/ross/CHADBuilder/CHAD.db")
    yield con
    # Close database when complete
    con.close()
