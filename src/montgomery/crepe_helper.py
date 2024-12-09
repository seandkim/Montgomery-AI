from typing import List
import crepe
import librosa
import numpy as np
import time
import matplotlib.pyplot as plt

from montgomery.helper import print_verbose


class PitchInfo:
    def __init__(
        self, timestamp, duration, frequency, confidence, shift_by_half_note: int = 0
    ):
        self.timestamp = timestamp
        self.duration = duration
        midi = librosa.hz_to_midi(frequency) + shift_by_half_note
        self.note = librosa.midi_to_note(midi.astype(int))
        self.confidences: List[float] = [confidence]

    def __repr__(self):
        return f"timestamp: {self.timestamp:.2f}s, duration: {self.duration:.2f}s, Note: {self.note} (Avg Conf: {np.mean(self.confidences):.2f})"

    # merge pitch_info if there are same note
    def merge(self, other_pitch_info):
        if self.note != other_pitch_info.note:
            raise ValueError("Cannot merge PitchInfo with different notes")
        else:
            self.duration += other_pitch_info.duration
            self.confidences.extend(other_pitch_info.confidences)


# merge consecutive pitch info if same note
def smooth_pitch_infos(pitch_infos: List[PitchInfo], min_duration_ms=50):
    if not pitch_infos:
        return []

    def merge(pitch_infos1: List[PitchInfo]):
        merged_pitch_infos = [pitch_infos1[0]]
        for pitch_info in pitch_infos1[1:]:
            last_pitch_info = merged_pitch_infos[-1]
            if last_pitch_info.note == pitch_info.note:
                last_pitch_info.merge(pitch_info)
            else:
                merged_pitch_infos.append(pitch_info)
        return merged_pitch_infos

    smoothed = pitch_infos
    for _ in range(2):
        smoothed = merge(smoothed)
        smoothed = [p for p in smoothed if p.duration * 1000 > min_duration_ms]

    return smoothed


# Perform pitch prediction
def test_benmark(y: np.ndarray, sample_rate: float):
    print("Benchmark test for processing mp3 audio for 10 seconds")
    capacities = ["tiny", "small", "medium", "large", "full"]
    for c in capacities:
        start_time = time.time()
        crepe.predict(
            y, sample_rate, model_capacity=c
        )  # Options: 'tiny', 'small', 'medium', 'large', 'full'
        duration = time.time() - start_time
        print(f"Model capacity: {c}, Duration: {duration:.2f} seconds")


def run_crepe(
    file: str,
    shift_by_half_note: int = 0,
    model_capacity: str = "medium",
    offset_seconds: int = 0,
    duration_seconds: int = 5,
    confidence_threshold: float = 0.7,
) -> List[PitchInfo]:
    y, sample_rate = librosa.load(
        file, sr=16000, offset=offset_seconds, duration=duration_seconds
    )  # crepe works best with 16k
    start_time = time.time()
    timestamps, frequencies, confidences, activation = crepe.predict(
        y, sample_rate, model_capacity=model_capacity
    )
    print_verbose(
        f"CREPE finished: capacity={model_capacity}, total_audio_length={duration_seconds}, duration={time.time() - start_time:.2f}s"
    )

    mask = confidences > confidence_threshold
    timestamps = timestamps[mask]
    frequencies = frequencies[mask]
    confidences = confidences[mask]
    pitch_infos = []
    for idx in range(len(timestamps) - 1):
        timestamp = timestamps[idx]
        duration = timestamps[idx + 1] - timestamp
        pitch_infos.append(
            PitchInfo(
                timestamp,
                duration,
                frequencies[idx],
                confidences[idx],
                shift_by_half_note=shift_by_half_note,
            )
        )

    return smooth_pitch_infos(pitch_infos)


if __name__ == "__main__":
    file = "files/sweetchild/audio.mp3"
    pitch_infos = run_crepe(file, shift_by_half_note=1)
    for pitch_info in pitch_infos:
        print(pitch_info)
