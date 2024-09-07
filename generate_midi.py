import torch
import pretty_midi


song_tensor = torch.load("generated_song.pt")

tempo = 120

quad_croche_duration = 1 / tempo / 16

bar_duration_seconds = 1 / tempo * 4

notes = []

last_start = 0

for i in range(song_tensor.shape[1]):
    velocity = song_tensor[0, i, 3].item() - 1
    relative_start = song_tensor[0, i, 1].item() - 1
    duration = song_tensor[0, i, 2].item() - 1
    pitch_class = song_tensor[0, i, 0].item() - 1

    start_seconds = relative_start * quad_croche_duration
    if start_seconds < last_start:
        start_seconds += bar_duration_seconds

    end_seconds = start_seconds + (duration * quad_croche_duration)

    last_start = start_seconds
    
    notes.append(pretty_midi.Note(
        velocity=velocity,
        pitch=pitch_class,
        start=start_seconds,
        end=end_seconds
    ))

instru = pretty_midi.Instrument(0)
instru.notes = notes

res = pretty_midi.PrettyMIDI(initial_tempo=tempo)
res.instruments = [instru]
res.write('result.mid')