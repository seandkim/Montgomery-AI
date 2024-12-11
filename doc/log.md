# November

- Difficult to piece the libraries together
- folder structure, package reference nightmare

# December

## 3rd

- Mediapipe: had to tweak confidence parameter tuning to detect hand
- Colab: masks. Find rectangular mask?

## 6th

- PCA to crop and rotate the image based on mask
- canny edge: had to skip blurring because fret bar would not be detected

## 7th

- rotating hand coordinate was a pain... => create your own class

## 8th

- librosa: lots of noise, overtone
- multiple notes at the same time is tough
- crepe: cnn solution

## 9th

- 0th fret is hard to catch
- hard thing is we don't know which strings being played
- piano might be easier (z-index to detect if keys are pressed)


```shell
Benchmark test for processing mp3 audio for 10 seconds
32/32 ━━━━━━━━━━━━━━━━━━━━ 1s 15ms/step
Model capacity: tiny, Duration: 5.61 seconds
32/32 ━━━━━━━━━━━━━━━━━━━━ 1s 32ms/step 
Model capacity: small, Duration: 1.20 seconds
32/32 ━━━━━━━━━━━━━━━━━━━━ 2s 71ms/step 
Model capacity: medium, Duration: 2.46 seconds
32/32 ━━━━━━━━━━━━━━━━━━━━ 4s 138ms/step
Model capacity: large, Duration: 4.63 seconds
32/32 ━━━━━━━━━━━━━━━━━━━━ 8s 232ms/step
Model capacity: full, Duration: 7.67 seconds
```

```timestamp: 0.43s, duration: 0.42s, Note: A4 (Avg Conf: 0.55)
timestamp: 0.85s, duration: 0.53s, Note: C♯4 (Avg Conf: 0.53)
timestamp: 1.38s, duration: 0.50s, Note: A4 (Avg Conf: 0.55)
timestamp: 1.88s, duration: 0.01s, Note: C♯4 (Avg Conf: 0.61)
timestamp: 1.89s, duration: 0.23s, Note: D4 (Avg Conf: 0.90)
timestamp: 2.12s, duration: 0.02s, Note: C♯4 (Avg Conf: 0.82)
timestamp: 2.14s, duration: 0.16s, Note: D4 (Avg Conf: 0.82)
timestamp: 2.30s, duration: 0.02s, Note: D5 (Avg Conf: 0.88)
timestamp: 2.32s, duration: 0.07s, Note: C♯5 (Avg Conf: 0.84)
timestamp: 2.39s, duration: 0.21s, Note: A4 (Avg Conf: 0.91)
timestamp: 2.60s, duration: 0.02s, Note: G♯4 (Avg Conf: 0.72)
timestamp: 2.62s, duration: 0.06s, Note: G4 (Avg Conf: 0.94)
timestamp: 2.68s, duration: 0.02s, Note: F♯4 (Avg Conf: 0.94)
timestamp: 2.70s, duration: 0.15s, Note: G4 (Avg Conf: 0.94)
timestamp: 2.85s, duration: 0.02s, Note: F♯4 (Avg Conf: 0.88)
timestamp: 2.87s, duration: 0.21s, Note: G4 (Avg Conf: 0.94)
timestamp: 3.08s, duration: 0.02s, Note: F♯4 (Avg Conf: 0.63)
timestamp: 3.10s, duration: 0.23s, Note: A4 (Avg Conf: 0.85)
timestamp: 3.33s, duration: 0.18s, Note: F♯5 (Avg Conf: 0.91)
timestamp: 3.51s, duration: 0.05s, Note: F5 (Avg Conf: 0.92)
timestamp: 3.56s, duration: 0.25s, Note: A4 (Avg Conf: 0.88)
timestamp: 3.81s, duration: 0.26s, Note: D4 (Avg Conf: 0.86)
timestamp: 4.07s, duration: 0.16s, Note: D5 (Avg Conf: 0.76)
timestamp: 4.23s, duration: 0.06s, Note: C♯5 (Avg Conf: 0.82)
timestamp: 4.29s, duration: 0.22s, Note: A4 (Avg Conf: 0.89)
timestamp: 4.51s, duration: 0.02s, Note: G♯4 (Avg Conf: 0.50)
timestamp: 4.53s, duration: 0.06s, Note: G4 (Avg Conf: 0.93)
timestamp: 4.59s, duration: 0.01s, Note: F♯4 (Avg Conf: 0.95)
timestamp: 4.60s, duration: 0.16s, Note: G4 (Avg Conf: 0.94)
timestamp: 4.76s, duration: 0.01s, Note: F♯4 (Avg Conf: 0.89)
timestamp: 4.77s, duration: 0.19s, Note: G4 (Avg Conf: 0.94)
timestamp: 4.96s, duration: 0.03s, Note: G5 (Avg Conf: 0.80)
```

```

timestamp: 1.90s, duration: 0.38s, Note: D4 (Avg Conf: 0.87)
timestamp: 2.32s, duration: 0.07s, Note: C♯5 (Avg Conf: 0.89)
timestamp: 2.39s, duration: 0.21s, Note: A4 (Avg Conf: 0.91)
timestamp: 2.62s, duration: 0.45s, Note: G4 (Avg Conf: 0.94)
timestamp: 3.11s, duration: 0.23s, Note: A4 (Avg Conf: 0.89)
timestamp: 3.34s, duration: 0.17s, Note: F♯5 (Avg Conf: 0.93)
timestamp: 3.51s, duration: 0.06s, Note: F5 (Avg Conf: 0.92)
timestamp: 3.57s, duration: 0.25s, Note: A4 (Avg Conf: 0.91)
timestamp: 3.82s, duration: 0.27s, Note: D4 (Avg Conf: 0.87)
timestamp: 4.09s, duration: 0.14s, Note: D5 (Avg Conf: 0.81)
timestamp: 4.23s, duration: 0.07s, Note: C♯5 (Avg Conf: 0.82)
timestamp: 4.30s, duration: 0.23s, Note: A4 (Avg Conf: 0.91)
timestamp: 4.53s, duration: 0.41s, Note: G4 (Avg Conf: 0.94)
```