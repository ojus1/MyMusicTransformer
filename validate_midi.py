import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from miditok import MIDILike
from miditoolkit import MidiFile
from pathlib import Path
from sklearn.model_selection import train_test_split

# Creates the tokenizer and loads a MIDI
in_root = "data/all"

def process_one(f):
    tokenizer = MIDILike()
    try:
        in_path = os.path.join(in_root, f)
        mid = MidiFile(in_path)
        tokens = tokenizer(mid, apply_bpe_if_possible=False)
        tokens[0]
        return True
    except Exception as e:
        print(e)
        return False

input_files = os.listdir(in_root)
valids = Parallel(n_jobs=-1, prefer='processes')(delayed(process_one)(f) for f in tqdm(input_files))

df = pd.DataFrame(zip(input_files, valids), columns=["file", "is_valid"])
df = df.loc[df.is_valid].reset_index(drop=True).drop(columns=["is_valid"])
df["file"] = df["file"].apply(lambda x: os.path.join(in_root, x))
train, test = train_test_split(df, train_size=0.9, random_state=123)
print("#Total", len(input_files))
print("#Rejected", len(input_files) - len(df))
print("#Train", len(train))
print("#Test", len(test))
train.to_csv("dump/train.csv", index=False)
test.to_csv("dump/test.csv", index=False)
