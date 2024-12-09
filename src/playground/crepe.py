import crepe
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file using librosa
audio_path = "files/sweetchild/audio.mp3"
duration_seconds = 5
y, sr = librosa.load(
    audio_path, sr=16000, duration=duration_seconds
)  # CREPE works best with 16kHz

# Perform pitch prediction
time, frequency, confidence, activation = crepe.predict(
    y, sr, model_capacity="tiny"
)  # Options: 'tiny', 'small', 'medium', 'large', 'full'

# Filter predictions based on confidence
confidence_threshold = 0.5
mask = confidence > confidence_threshold
time = time[mask]
frequency = frequency[mask]
confidence = confidence[mask]

# Convert frequencies to MIDI notes and note names
midi = librosa.hz_to_midi(frequency) + 1
note_names = librosa.midi_to_note(midi.astype(int))  # shift half note

# Print detected notes
for t, f, n, c in zip(time, frequency, note_names, confidence):
    print(f"Time: {t:.2f}s, Frequency: {f:.2f}Hz, Note: {n}, Confidence: {c:.2f}")

# Optional: Plotting the pitch over time
plt.figure(figsize=(14, 6))
plt.plot(time, frequency, label="Estimated Pitch (Hz)", color="b")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("CREPE Pitch Estimation")
plt.legend()
plt.show(block=True)
