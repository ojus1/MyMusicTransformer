import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from miditok import MIDILike
from symusic import Score
from sklearn.model_selection import train_test_split
from glob import glob

tokenizer = MIDILike()

def process_one(f):
    try:
        in_path = f
        mid = Score(in_path)
        tokens = tokenizer(mid)
        tokens[0]
        return True
    except Exception as e:
        print(e)
        return False

# Creates the tokenizer and loads a MIDI
in_root = "scrape/data/**/*mid*"
input_files = glob(in_root) + glob("scrape/data/**/*MID*")

valids = Parallel(n_jobs=128, prefer='threads')(delayed(process_one)(f) for f in tqdm(input_files))

df = pd.DataFrame(zip(input_files, valids), columns=["file", "is_valid"])
df = df.loc[df.is_valid].reset_index(drop=True).drop(columns=["is_valid"])
train, test = train_test_split(df, train_size=0.9, random_state=123)
print("#Total", len(input_files))
print("#Rejected", len(input_files) - len(df))
print("#Train", len(train))
print("#Test", len(test))
train, valid = train_test_split(train, train_size=0.95, random_state=123)
train.to_csv("dump/train.csv", index=False)
valid.to_csv("dump/valid.csv", index=False)
test.to_csv("dump/test.csv", index=False)
