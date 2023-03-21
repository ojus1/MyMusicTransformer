from mido import MidiFile
import os
from joblib import Parallel, delayed
from tqdm import tqdm
os.makedirs("data/all_validated", exist_ok=True)

in_root = "data/all"
out_root = "data/all_validated"

def process_one(f):
    try:
        in_root = "data/all"
        out_root = "data/all_validated"
        mid = MidiFile(os.path.join(in_root, f))

        max_len = 0
        max_len_track = None
        for i, track in enumerate(mid.tracks):
            if max_len < len(track):
                max_len = len(track)
                max_len_track = track

        new_mid = MidiFile()
        new_mid.tracks.append(max_len_track)
        new_mid.save(os.path.join(out_root, f))
    except:
        pass

# for f in tqdm(os.listdir(in_root)):

Parallel(n_jobs=-1, prefer='processes')(delayed(process_one)(f) for f in tqdm(os.listdir(in_root)))