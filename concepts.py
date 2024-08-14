"""
Has different functions to calculate high level concepts from a list of musical notes
"""

from typing import List, Callable, Tuple
from tokenizer import NoteToken
from dataclasses import dataclass


def calculate_average_velocity_per_bar(note_tokens: List[NoteToken]) -> List[NoteToken]:
    # Group notes by the bar (using previous_downbeat as the identifier)
    bars = {}
    for note in note_tokens:
        bars[note.previous_downbeat].append(note)

    # Calculate average velocity for each bar and assign it to notes
    for bar_notes in bars.values():
        average_velocity = sum(note.velocity for note in bar_notes) / len(bar_notes)
        for note in bar_notes:
            note.concepts['bar_velocity'] = average_velocity
    
    return note_tokens

def calculate_average_absolute_pitch_per_bar(note_tokens: List[NoteToken]) -> List[NoteToken]:
    # Group notes by the bar (using previous_downbeat as the identifier)
    bars = {}
    for note in note_tokens:
        bars[note.previous_downbeat].append(note)

    # Calculate average pitch for each bar and assign it to notes
    for bar_notes in bars.values():
        average_pitch = sum(note.pitch + note.octave*12 for note in bar_notes) / len(bar_notes)
        for note in bar_notes:
            note.concepts['bar_pitch'] = average_pitch
    
    return note_tokens


@dataclass
class Concept:
    name: str
    calculate: Callable[[List[NoteToken]], List[NoteToken]]
    range: Tuple[float, float]
    is_discrete: bool = False
    field_name: str


CONCEPTS = [
    Concept("Average Velocity per Bar", calculate_average_velocity_per_bar, (0, 127), False, "bar_velocity"),
    Concept("Average Absolute Pitch per Bar", calculate_average_absolute_pitch_per_bar, (0, 127), False, "bar_pitch")
]
