import librosa
import numpy as np
import math


def hz_to_note_name(frequency, shift_by_half_note: int = 0):
    # Reference A4 = 440 Hz
    A4_frequency = 440.0
    # Chromatic scale notes starting from C
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Calculate the number of semitones from A4
    semitones_from_A4 = round(12 * math.log2(frequency / A4_frequency))

    # Determine the octave and note name
    note_index = (semitones_from_A4 + 9) % 12  # A4 is index 9
    octave = (semitones_from_A4 + 9) // 12 + 4  # A4 is in octave 4

    note_index = (note_index + shift_by_half_note) % 12
    note_name = note_names[note_index]
    return f"{note_name}{octave}"


def note_array_to_tuple_of_occurrence(notes):
    result = []
    if not notes:  # Handle empty list
        return result

    current_note = notes[0]
    count = 1

    for note in notes[1:]:
        if note == current_note:
            count += 1
        else:
            result.append((current_note, count))
            current_note = note
            count = 1
    # Add the last note and its count
    result.append((current_note, count))
    return result


# Load the audio file (mono)
audio_path = "files/sweetchild/audio.mp3"
y, sr = librosa.load(audio_path, sr=None)

# Define pitch range (adjust based on your expected pitch range)
fmin = librosa.note_to_hz("C2")  # ~65 Hz
fmax = librosa.note_to_hz("C7")  # ~2093 Hz

# Run YIN pitch detection
f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)

# Create a time array for the pitch values
times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)

# Example: Get pitch at a particular time (e.g., time_requested = 2.5 seconds)
time_requested = 2.5
# Find the closest frame index
frame_index = np.argmin(np.abs(times - time_requested))
pitch_at_time = f0[frame_index]
print(
    f"Pitch at {time_requested:.2f} s: {pitch_at_time:.2f} Hz ({hz_to_note_name(pitch_at_time)})"
)

# Example: Get pitch over an interval (e.g., from 2.0 to 3.0 seconds)
start_time = 1.0
end_time = 5.0

# Find frame indices for the interval
start_frame = np.searchsorted(times, start_time)
end_frame = np.searchsorted(times, end_time)

interval_pitches = f0[start_frame:end_frame]
note_names = [hz_to_note_name(h, shift_by_half_note=1) for h in interval_pitches]
note_occurrences = note_array_to_tuple_of_occurrence(note_names)
note_occurrences = [(note, occ) for (note, occ) in note_occurrences if occ >= 5]
for note, occurrence in note_occurrences:
    print(note, occurrence)

print(note_occurrences)

#
# average_pitch = np.mean(
#     interval_pitches[~np.isnan(interval_pitches)]
# )  # ignore NaNs if any
# print(
#     f"Average pitch between {start_time:.2f} s and {end_time:.2f} s: {average_pitch:.2f} Hz"
# )
