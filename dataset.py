from typing import List
from torch.utils.data import Dataset
import torch
from tokenizer import tokenize
import pretty_midi
from pathlib import Path
from tqdm import tqdm

def get_file_paths(dataset_path):
    result = Path(dataset_path).glob("**/*.mid")
    result = [str(x) for x in result]
    return result


    
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def safe_parse_midi(path):
    try:
        return tokenize(pretty_midi.PrettyMIDI(path))
    except:
        return None

class MIDIRepresentationDataset(Dataset):
    def __init__(self, midi_file_paths: List[str], max_length: int) -> None:
        super().__init__()

        # self.songs = [
        #     [x.to_vector() for x in safe_parse_midi(path) if x is not None]
        #     for path in tqdm(midi_file_paths[:50])
        # ]

        self.songs = []
        for path in tqdm(midi_file_paths[:50]):
            parsed_midi = safe_parse_midi(path)
            if parsed_midi is None:
                continue

            self.songs.append([x.to_vector() for x in parsed_midi])



        print("Max pitch", max([max([note[0] for note in song]) for song in self.songs]))
        print("Max start", max([max([note[1] for note in song]) for song in self.songs]))
        print("Max duration", max([max([note[2] for note in song]) for song in self.songs]))
        print("Max velocity", max([max([note[3] for note in song]) for song in self.songs]))
        print("Max instrument", max([max([note[4] for note in song]) for song in self.songs]))
        print("Min length", min([len(x) for x in self.songs]))

        self.max_length = max_length

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, index) -> torch.Tensor:
        input_ids = torch.tensor(self.songs[index][:self.max_length]).long()
        return {'input_ids': input_ids, 'attention_mask': torch.ones(input_ids.shape[:1]).long(), 'labels': input_ids.clone()}
