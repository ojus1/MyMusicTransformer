from miditok import MIDILike
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from pathlib import Path

# Creates the tokenizer and loads a MIDI
tokenizer = MIDILike()  # using the default parameters, read the documentation to customize your tokenizer
# midi = MidiFile('data/all/Zweig, Otto, Suite, Op.6, RXnBSWZbcUc.mid')

# # Converts MIDI to tokens, and back to a MIDI
# tokens = tokenizer(midi)  # automatically detects MIDIs and tokens before converting
# converted_back_midi = tokenizer(tokens, get_midi_programs(midi))  # PyTorch / Tensorflow / Numpy tensors supported
# print(tokens)

# Converts MIDI files to tokens saved as JSON files
midi_paths = list(Path('data/all/').glob('*.mid'))
data_augmentation_offsets = [2, 2, 1]  # data augmentation on 2 pitch octaves, 2 velocity and 1 duration values
tokenizer.tokenize_midi_dataset(midi_paths, Path('data/tokens_noBPE'),
                                data_augment_offsets=data_augmentation_offsets)

# Constructs the vocabulary with BPE, from the tokenized files
tokenizer.learn_bpe(
    vocab_size=500,
    tokens_paths=list(Path('data/tokens_noBPE').glob("*.json")),
    start_from_empty_voc=False,
)

# Converts the tokenized musics into tokens with BPE
tokenizer.apply_bpe_to_dataset(Path('data/tokens_noBPE'), Path('data/tokens_BPE'))
