from concepts import CONCEPTS
import numpy as np
import pretty_midi
from tqdm import tqdm

# def get_previous_downbeat(note: pretty_midi.Note, midi: pretty_midi.PrettyMIDI):
#     for i in range(len(midi.get_downbeats())):
#         if note.start < midi.get_downbeats()[i]:
#             return midi.get_downbeats()[i-1]
#     return midi.get_downbeats()[-1]

def get_64th_note_duration(midi: pretty_midi.PrettyMIDI):
    if midi.time_signature_changes[0].numerator % 3 == 0 and midi.time_signature_changes[0].numerator > 3:
        return (midi.get_beats()[2] - midi.get_beats()[1]) / 24        
    return (midi.get_beats()[2] - midi.get_beats()[1]) / 16

def get_note_duration(note: pretty_midi.Note, tick: float):
    return (note.end - note.start) // tick

def get_note_position_in_bar(note: pretty_midi.Note, tick: float, previous_downbeat: float):
    time_within_bar = note.start - previous_downbeat
    if time_within_bar % tick > tick / 2:
        return time_within_bar // tick + 1
    return time_within_bar // tick

class NoteToken:
    def __init__(self, note: pretty_midi.Note, instrument: int, time_tick: float, previous_downbeat: float):
        self.pitch = note.pitch % 12
        # TODO: Understand why we get negative values here
        self.start = max(get_note_position_in_bar(note, time_tick, previous_downbeat), 0)
        self.duration = get_note_duration(note, time_tick)
        self.velocity = note.velocity
        self.instrument = instrument

        self.octave = note.pitch // 12
        self.absolute_start = note.start

        # Can be used to identify the bars
        self.previous_downbeat = previous_downbeat

        self.concepts = {}
        
    # def __eq__(self, other):
    #     return self.pitch == other.pitch and self.position_in_bar == other.position_in_bar

    # def __hash__(self):
    #     return hash((self.pitch, self.position_in_bar))

    def __repr__(self):
        return f"Pitch: {self.pitch}, Position in bar: {self.start}, Duration: {self.duration}, Velocity: {self.velocity}, Instr: {self.instrument}"
    
    def to_vector(self):
        # Shift values by one to account for padding tokens with 0
        return np.array([
            self.pitch+1,
            self.start+1,
            self.duration+1,
            self.velocity+1,
            self.instrument+1
        ])
    
    def concepts_to_vector(self):
        return np.array([self.concepts[concept.field_name] for concept in CONCEPTS])

    @classmethod
    def pad_token(cls):
        return np.array([0]*5)

    
def tokenize(midi: pretty_midi.PrettyMIDI):
    result = []
    tick = get_64th_note_duration(midi)
    downbeats = midi.get_downbeats()
    for ind, track in enumerate(midi.instruments):
        current_downbeat_ind = 0
        for note in track.notes:
            if current_downbeat_ind < len(downbeats) - 1 and note.start > downbeats[current_downbeat_ind+1]:
                current_downbeat_ind += 1

            result.append(NoteToken(
                note, track.program, tick, downbeats[current_downbeat_ind]
            ))

    result.sort(key=lambda x: (x.absolute_start, x.octave, x.pitch, x.instrument))

    return result
