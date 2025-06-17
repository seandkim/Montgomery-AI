# Montgomery

## Setup

1. Download Segment Anything 2.1
2. Download MediaPipe
3. `conda activate montgomery`
4. `python -m montgomery.main`

## To Run

- `pip install -e .`: one time only
- `python -m montgomery.main`

## Version

- python=3.12
- torch=2.5.1
- numpy=1.26.4

conda create -n montgomery python=3.12 numpy=1.26.4 matplotlib pytorch=2.5.1 torchvision torchaudio -c pytorch

## References

- [Facebook SAM2](https://github.com/facebookresearch/sam2)

### Python help

Modules
- https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#src-layout
- https://github.com/pypa/setuptools/issues/4248

## Baseline

- ChatGPT: "How do you play the intro riff for "Sunshine of your love" by Cream? Give me the tab score"

## Test videos

Youtube videos were downloaded via "https://cnvmp3.com/", "https://yt1d.com/en12/". The link can be found in `input.json`
