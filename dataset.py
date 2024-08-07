from typing import List
from torch.utils.data import Dataset
import torch
from tokenizer import tokenize, NoteToken
import pretty_midi
from pathlib import Path
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import traceback
import os
import sys
from config import DATASET_PATH
import wandb
from logger import wandb_logger


def get_file_paths(dataset_path):
    result = Path(dataset_path).glob("**/*.mid")
    result = [str(x) for x in result]
    return result


def clear_cache(dataset_path):
    result = Path(dataset_path).glob("**/*.cache")
    result = [str(x) for x in result]
    for f in result:
        os.remove(f)
    print("Cache cleared!")
    return result


def validate_sorting(song):
    # Test 10 times the sorting randomly
    for _ in range(10):
        i = np.random.randint(0, len(song)-2)
        j = np.random.randint(i+1, len(song)-1)
        x, y = song[i], song[j]
        try:
            if x.absolute_start == y.absolute_start:
                if x.octave * 12 + x.pitch == y.octave*12 + y.pitch:
                    assert x.instrument < y.instrument, "Broken Instrument sorting"
                else:
                    assert x.octave * 12 + x.pitch < y.octave*12 + y.pitch, "Broken Pitch sorting"
            else:
                assert x.absolute_start < y.absolute_start, "Broken Time sorting"
        except Exception as e:
            print("Failed Sorting Validation:", e)

    
def collate_fn(batch):
    #  TODO: Add padding here
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def safe_parse_midi(path):
    try:
        cache_path = path[:-3] + "cache"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        result = tokenize(pretty_midi.PrettyMIDI(path))
        validate_sorting(result)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        return result
    except Exception as e:
        # print(e)
        return None

class MIDIRepresentationDataset(Dataset):
    def __init__(self, midi_file_paths: List[str], max_length: int, piano_only=False, padding=False) -> None:
        super().__init__()
        self.max_length = max_length

        self.n_instances = 0
        # self.songs = [
        #     [x.to_vector() for x in safe_parse_midi(path) if x is not None]
        #     for path in tqdm(midi_file_paths[:50])
        # ]

        # midi_file_paths = midi_file_paths[:100]

        self.songs = np.memmap("data.tmp", shape=(len(midi_file_paths), max_length, 5), mode="w+", dtype=np.uint16)
        self.note_octaves = np.memmap("octaves.tmp", shape=(len(midi_file_paths), max_length), mode="w+")
        self.attention_masks = np.memmap("attention_masks.tmp", shape=(len(midi_file_paths), max_length), mode="w+")
        self.song_lengths = []
        print(f"Found {len(midi_file_paths)} total paths")
        clipped_start_or_duration = 0
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(safe_parse_midi, path) for path in midi_file_paths]

            for f in tqdm(as_completed(futures), total=len(midi_file_paths)):
                parsed_midi = f.result()
                if parsed_midi is None:
                    continue

                if piano_only:
                    parsed_midi = [x for x in parsed_midi if x.instrument == 0]
                    # Skip if we want a piano dataset and the song does not contain piano
                    if not len(parsed_midi):
                        continue


                # TODO: This is also a temporary solution until we have bins
                for x in parsed_midi:
                    if x.start > 1000:
                        x.start = 1000
                        clipped_start_or_duration+=1
                    if x.duration > 1000:
                        x.duration = 1000
                        clipped_start_or_duration+=1

                song = [x.to_vector() for x in parsed_midi]

                # Degenerate dataset instance, skip
                if not len(song):
                    continue
                non_clipped_song_length = len(song)
                song = song[:self.max_length]
                original_song_length = len(song)
                self.song_lengths.append(non_clipped_song_length)
                song += [NoteToken.pad_token()]*(self.max_length - len(song))

                octaves = [x.octave for x in parsed_midi][:self.max_length]
                octaves += [-1]*(self.max_length - len(octaves))

                attention_mask = np.append(
                    np.ones(original_song_length),
                    np.zeros(self.max_length - original_song_length)
                )

                self.songs[self.n_instances, :, :] = song
                self.note_octaves[self.n_instances, :] = octaves
                self.attention_masks[self.n_instances, :] = attention_mask

                self.n_instances += 1
        # for path in tqdm(midi_file_paths):
        #     parsed_midi = safe_parse_midi(path)
        print("Number of instances", self.n_instances)
        print("Tokens with clipped Start or Duration", clipped_start_or_duration)

        # TODO: We might log this later on wandb
        # instrument_usage = {}
        # for x in self.songs[:self.n_instances]:
        #     used_instruments = set()
        #     for note in x:
        #         used_instruments.add(note[4])
        #     for instr in used_instruments:
        #         if instr not in instrument_usage:
        #             instrument_usage[instr] = 0
        #         instrument_usage[instr] += 1
        # print("Most used instrument")
        # for instr, count in sorted(instrument_usage.items(), key=lambda x: (x[1],x[0]), reverse=True)[:10]:
        #     print("Instrument", instr-1, "used in ", count, "songs")
        print()
        
        print("Max pitch", max([max([note[0] for note in song]) for song in self.songs[:self.n_instances]]))
        print("Max start", max([max([note[1] for note in song]) for song in self.songs[:self.n_instances]]))
        print("Max duration", max([max([note[2] for note in song]) for song in self.songs[:self.n_instances]]))
        print("Max velocity", max([max([note[3] for note in song]) for song in self.songs[:self.n_instances]]))
        print("Max instrument", max([max([note[4] for note in song]) for song in self.songs[:self.n_instances]]))
        print("Min length", min([len(x) for x in self.songs[:self.n_instances]]))

        assert not np.isnan(np.sum(self.songs[:self.n_instances])), "Some songs have NaN!"
        assert not np.any(np.isinf(self.songs[:self.n_instances])), "Some songs have Inf!"

        self._log_dataset_metadata()

    def _log_dataset_metadata(self):
        # Log the original song lengths
        table = wandb.Table(data=[[x] for x in self.song_lengths], columns=["song_length"])
        wandb_logger.experiment.log({'song_length': wandb.plot.histogram(table, "song_length",
                title="Song Length")})


    def __len__(self):
        return self.n_instances

    def __getitem__(self, index) -> torch.Tensor:
        input_ids = torch.tensor(self.songs[index]).long()
        return {'input_ids': input_ids, 'attention_mask': torch.tensor(self.attention_masks[index]).long(), 'labels': input_ids.clone()}

    def __del__(self):
        self.songs._mmap.close()
        self.note_octaves._mmap.close()
        self.attention_masks._mmap.close()
        os.remove("data.tmp")
        os.remove("attention_masks.tmp")
        os.remove("octaves.tmp")

if __name__ == "__main__":
    if sys.argv[1] == "clear-cache":
        clear_cache(DATASET_PATH)
