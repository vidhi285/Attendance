# test_db.py
from database import DatabaseManager

db = DatabaseManager()
ok = db.connect()
print("Connected?", ok)

if ok:
    try:
        rows = db.execute_query("SELECT 1 as val", fetch=True)
        print("Query result:", rows)
    except Exception as e:
        print("Query failed:", e)
    finally:
        db.disconnect()
