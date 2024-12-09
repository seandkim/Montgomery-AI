import numpy as np
import os

from montgomery import helper

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
from moviepy import VideoFileClip


def load_video(file_path: str) -> VideoFileClip:
    return VideoFileClip(file_path)


def get_frame(video: VideoFileClip, timestamp: float) -> np.ndarray:
    return


# Example usage:
if __name__ == "__main__":
    video = load_video("files/sweetchild/video.mp4")
    frame_rgb = video.get_frame(10)
    helper.show_image(frame_rgb)
