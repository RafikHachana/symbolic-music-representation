import torch
import pretty_midi


song_tensor = torch.load("generated_song.pt")

tempo = 120

quad_croche_duration = 60 / tempo / 16

bar_duration_seconds = 60 / tempo * 4

print(bar_duration_seconds)

notes = []

last_start = 0

for i in range(song_tensor.shape[1]):
    velocity = song_tensor[0, i, 3].item() - 1
    relative_start = song_tensor[0, i, 1].item() - 1
    duration = song_tensor[0, i, 2].item() - 1
    pitch_class = song_tensor[0, i, 0].item() - 1

    print("Pitch", pitch_class, "Start", relative_start, "Duration", duration, "Velocity", velocity)

    try:
        assert velocity < 127 and velocity > 0
        assert pitch_class > 0
        assert velocity < 127 and velocity > 0
    except:
        continue

    start_seconds = relative_start * quad_croche_duration
    while start_seconds < last_start:
        start_seconds += bar_duration_seconds

    end_seconds = start_seconds + (duration * quad_croche_duration)

    last_start = start_seconds
    
    notes.append(pretty_midi.Note(
        velocity=velocity,
        pitch=pitch_class+48,
        start=start_seconds,
        end=end_seconds
    ))

# for n in notes:
#     print("Start", n.start, "pitch", n.pitch)

# exit()

instru = pretty_midi.Instrument(0)
instru.notes = notes

res = pretty_midi.PrettyMIDI(initial_tempo=tempo)
res.instruments = [instru]
# res.remove_invalid_notes()
res.write('result.mid')
# res.fluidsynth('result.wav')