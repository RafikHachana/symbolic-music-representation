"""
Has different functions to calculate high level concepts from a list of musical notes
"""

from typing import List, Callable, Tuple
from dataclasses import dataclass
import math
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

# Dictionary mapping note names to semitone offsets from A4
NOTE_TO_SEMITONES = {
    'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4, 'F#': -3, 
    'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2
}

# Constants for Sethares's roughness model
ALPHA = 3.5
BETA = 5.75

# Function to convert note name to frequency
def note_to_frequency(note: str) -> float:
    # Parse the note name and octave (e.g., "C4", "D#5")
    if len(note) == 2:
        pitch, octave = note[0], int(note[1])
        accidental = ''
    elif len(note) == 3:
        pitch, accidental, octave = note[0], note[1], int(note[2])
    else:
        raise ValueError(f"Invalid note format: {note}")
    
    # Combine pitch and accidental if present (e.g., "C#", "D")
    full_note = pitch + accidental
    
    # Get the semitone offset from A4
    semitone_offset = NOTE_TO_SEMITONES[full_note]
    
    # Calculate the distance from A4 (which is in octave 4)
    n = semitone_offset + (octave - 4) * 12
    
    # Calculate the frequency using the formula for equal temperament
    frequency = 440 * (2 ** (n / 12))
    
    return frequency

# Function to calculate roughness between two frequencies
def roughness(f1: float, f2: float) -> float:
    return math.exp(-ALPHA * abs(f2 - f1)) * (1 - math.exp(-BETA * abs(f2 - f1)))

# Function to calculate total tonal tension for each bar
def calculate_tonal_tension_per_bar(note_tokens: List) -> List:
    # Step 1: Group notes into bars (this part assumes a group_bars function is defined elsewhere)
    bars = group_bars(note_tokens)
    
    # Step 2: Calculate tonal tension for each bar
    for bar_notes in bars.values():
        N = len(bar_notes)
        total_tension = 0.0
        
        # Step 3: Convert note pitches to frequencies and calculate roughness between each pair
        for i in range(N):
            f1 = note_to_frequency(bar_notes[i].pitch)  # Convert pitch to frequency
            for j in range(i + 1, N):
                f2 = note_to_frequency(bar_notes[j].pitch)  # Convert pitch to frequency
                total_tension += roughness(f1, f2)
        
        # Normalize the tension if there are multiple notes in the bar
        if N > 1:
            total_tension /= (N * (N - 1)) / 2  # Normalizing by the number of note pairs
        
        # Step 4: Assign the tonal tension to each note in the bar
        for note in bar_notes:
            note.concepts['bar_tonal_tension'] = total_tension
    
    # Return the modified note tokens with the assigned tonal tension
    return note_tokens


from typing import List

# Example beat hierarchy for a 4/4 time signature (stronger beats get higher weights)
BEAT_WEIGHTS = {
    1: 4,  # Strong downbeat
    2: 1,  # Weak beat
    3: 3,  # Medium strong beat
    4: 1   # Weak beat
}

# Function to calculate syncopation index per bar and assign it to note tokens
def calculate_syncopation_index_per_bar(note_tokens: List) -> List:
    bars = group_bars(note_tokens)

    # Calculate syncopation index for each bar
    for bar_notes in bars.values():
        syncopation_index = 0.0

        # Loop through the notes in the bar
        for i in range(len(bar_notes) - 1):
            current_note = bar_notes[i]
            next_note = bar_notes[i + 1]
            
            # Check if current note is syncopated (occurs on a weak beat but tied to a stronger one)
            if BEAT_WEIGHTS[current_note.beat] < BEAT_WEIGHTS[next_note.beat]:
                syncopation_index += BEAT_WEIGHTS[next_note.beat] - BEAT_WEIGHTS[current_note.beat]

        # Assign the syncopation index to each note in the bar
        for note in bar_notes:
            note.concepts['bar_syncopation_index'] = syncopation_index

    return note_tokens


# Function to calculate the normalized pairwise variability index (nPVI) per bar
def calculate_nPVI_per_bar(note_tokens: List) -> List:
    bars = group_bars(note_tokens)

    # Calculate nPVI for each bar
    for bar_notes in bars.values():
        note_durations = [note.duration for note in bar_notes]
        m = len(note_durations)
        if m < 2:
            nPVI = 0.0  # nPVI is not defined for fewer than 2 notes
        else:
            nPVI = 0.0
            for k in range(m - 1):
                d1 = note_durations[k]
                d2 = note_durations[k + 1]
                nPVI += abs(d2 - d1) / ((d2 + d1) / 2)
            nPVI = (nPVI / (m - 1)) * 100
        
        # Assign the nPVI value to each note in the bar
        for note in bar_notes:
            note.concepts['bar_nPVI'] = nPVI

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
    Concept("Average Absolute Pitch per Bar", calculate_average_absolute_pitch_per_bar, (0, 127), "bar_pitch"),
    Concept("Tonal Tension per Bar", calculate_tonal_tension_per_bar, (0, 1), "bar_tonal_tension"),
    Concept("Syncopation Index per Bar", calculate_syncopation_index_per_bar, (0, 1), "bar_syncopation_index"),
    Concept("Normalized Pairwise Variability Index (nPVI) per Bar", calculate_nPVI_per_bar, (0, 100), "bar_nPVI")
]
