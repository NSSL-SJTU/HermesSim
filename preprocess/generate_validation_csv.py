import pandas as pd
from os.path import join

DB_DIR = "dbs"
INPUT = join(DB_DIR, "Dataset-1/validation_Dataset-1.csv")
OUTPUT = join(DB_DIR, "Dataset-1/pairs/validation/validation_functions.csv")

df = pd.read_csv(INPUT, index_col = 0)

name2groupdid = {}
counter = 0
groups = []

for idx, r in df.iterrows():
    fname = r["func_name"]
    if not fname in name2groupdid:
        name2groupdid[fname] = counter
        counter += 1
    groups.append(name2groupdid[fname])

df = df[["idb_path","fva"]]
df.insert(2, "group", groups)

df.to_csv(OUTPUT)
