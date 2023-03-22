from miditok import MIDILike
from pathlib import Path
import pickle
import pandas as pd

tokenizer = MIDILike()  # using the default parameters, read the documentation to customize your tokenizer

# Converts MIDI files to tokens saved as JSON files
tr_midi_paths = pd.read_csv("dump/train.csv")['file'].to_list()
ts_midi_paths = pd.read_csv("dump/test.csv")['file'].to_list()
# data_augmentation_offsets = [2, 2, 1]  # data augmentation on 2 pitch octaves, 2 velocity and 1 duration values
# tokenizer.tokenize_midi_dataset(tr_midi_paths, Path('dump/train_noBPE'),
#                                 data_augment_offsets=data_augmentation_offsets)
# tokenizer.tokenize_midi_dataset(ts_midi_paths, Path('dump/test_noBPE'))

# Constructs the vocabulary with BPE, from the tokenized files
tokenizer.learn_bpe(
    vocab_size=512,
    tokens_paths=list(Path('dump/train_noBPE').glob("*.json")),
    start_from_empty_voc=False,
)

pickle.dump(tokenizer, open("dump/midilike_tokenizer.pkl", "wb"))
# Converts the tokenized musics into tokens with BPE
tokenizer.apply_bpe_to_dataset(Path('dump/train_noBPE'), Path('dump/train_BPE'))
tokenizer.apply_bpe_to_dataset(Path('dump/test_noBPE'), Path('dump/test_BPE'))
