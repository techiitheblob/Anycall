import sqlite3
import shutil
from pathlib import Path

# Delete from DB
conn = sqlite3.connect('anycall.db')
c = conn.cursor()
c.execute("DELETE FROM species WHERE species_id = 'duttaphrynus_melanostictus'")
conn.commit()
conn.close()
print('Deleted toad from DB')

# Delete processed audio
toad_dir = Path('data/processed/duttaphrynus_melanostictus')
if toad_dir.exists():
    shutil.rmtree(toad_dir)
    print('Deleted toad audio')
