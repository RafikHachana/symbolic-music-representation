"""
Has different functions to calculate high level concepts from a list of musical notes
"""

from typing import List, Callable, Tuple
from dataclasses import dataclass

def group_bars(note_tokens: List):
    # Group notes by the bar (using previous_downbeat as the identifier)
    bars = {}
    for note in note_tokens:
        if note.previous_downbeat not in bars:
            bars[note.previous_downbeat] = []
        bars[note.previous_downbeat].append(note)

    return bars


def calculate_average_velocity_per_bar(note_tokens: List) -> List:
    bars = group_bars(note_tokens)

    # Calculate average velocity for each bar and assign it to notes
    for bar_notes in bars.values():
        average_velocity = sum(note.velocity for note in bar_notes) / len(bar_notes)
        for note in bar_notes:
            note.concepts['bar_velocity'] = average_velocity
    
    return note_tokens

def calculate_average_absolute_pitch_per_bar(note_tokens: List) -> List:
    bars = group_bars(note_tokens)

    # Calculate average pitch for each bar and assign it to notes
    for bar_notes in bars.values():
        average_pitch = sum(note.pitch + note.octave*12 for note in bar_notes) / len(bar_notes)
        for note in bar_notes:
            note.concepts['bar_pitch'] = average_pitch
    
    return note_tokens


@dataclass
class Concept:
    name: str
    calculate: Callable[[List], List]
    range: Tuple[float, float]
    field_name: str
    is_discrete: bool = False


CONCEPTS = [
    Concept("Average Velocity per Bar", calculate_average_velocity_per_bar, (0, 127), "bar_velocity"),
    Concept("Average Absolute Pitch per Bar", calculate_average_absolute_pitch_per_bar, (0, 127), "bar_pitch")
]
