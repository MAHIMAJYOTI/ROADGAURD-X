"""
Generate a small synthetic driving scene video for local development.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

# Run this script once after cloning to generate the sample video.
# The generated file is excluded from git via .gitignore.
# Usage: python generate_sample.py
# Output: samples/street.mp4


def main() -> None:
    """Generate `samples/street.mp4` using a lightweight numpy drawing model."""

    # video specs
    FPS = 20
    WIDTH = 640
    HEIGHT = 360
    TOTAL_FRAMES = 300

    # drawing helpers / common numeric constants
    ORIGIN_X = 0
    ONE = 1.0
    TWO = 2.0
    PIXEL_MIN = 0
    PIXEL_MAX = 255
    DASH_SEGMENT_STEP = 2
    FILLED_THICKNESS = -1

    SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
    OUTPUT_PATH = SAMPLES_DIR / "street.mp4"

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # background layers
    SKY_END_Y = 144
    SKY_COLOR = (200, 180, 140)
    ROAD_COLOR = (100, 100, 100)
    HORIZON_Y = 144
    HORIZON_COLOR = (60, 60, 60)
    HORIZON_THICKNESS = 2

    # lane dash drawing
    VP_X_BASE = 320
    VP_X_AMPLITUDE = 60
    VP_OSC_PERIOD_FRAMES = 100
    VP_Y = 145

    LEFT_LANE_BOTTOM_X = 80
    RIGHT_LANE_BOTTOM_X = 560
    LANE_BOTTOM_Y = HEIGHT

    VP_INT_Y_START = VP_Y

    LANE_DASH_POINTS = 12
    LANE_DASH_SEGMENT_THICKNESS = 3
    LANE_DASH_COLOR = (255, 255, 255)

    # moving blobs
    BLOB_OFFSCREEN_Y = HEIGHT
    BLOB_RESET_Y = 160
    BLOB_X_BOUNCE_MIN = 50
    BLOB_X_BOUNCE_MAX = 580
    BLOB_X_RANDOM_MIN = 100.0
    BLOB_X_RANDOM_MAX = 500.0

    blobs = [
        {
            "x": 150.0,
            "y": 200.0,
            "w": 60,
            "h": 40,
            "vx": 0.4,
            "vy": 1.2,
            "color": (60, 60, 60),
        },
        {
            "x": 320.0,
            "y": 180.0,
            "w": 50,
            "h": 35,
            "vx": -0.3,
            "vy": 1.0,
            "color": (80, 75, 70),
        },
        {
            "x": 480.0,
            "y": 210.0,
            "w": 70,
            "h": 45,
            "vx": 0.5,
            "vy": 0.9,
            "color": (55, 60, 65),
        },
    ]

    # brightness + noise
    BRIGHTNESS_WOBBLE_AMPLITUDE = 0.25
    BRIGHTNESS_WOBBLE_PERIOD_FRAMES = 160
    NOISE_MEAN = 0.0
    NOISE_STD = 8.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, (WIDTH, HEIGHT))
    if not out.isOpened():
        print("Failed to create VideoWriter. Check that opencv-python is installed.")
        sys.exit(1)

    try:
        for frame_index in range(TOTAL_FRAMES):
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            # background fill
            frame[:SKY_END_Y, :, :] = SKY_COLOR
            frame[SKY_END_Y:, :, :] = ROAD_COLOR
            cv2.line(
                frame,
                (ORIGIN_X, HORIZON_Y),
                (WIDTH - 1, HORIZON_Y),
                HORIZON_COLOR,
                thickness=HORIZON_THICKNESS,
                lineType=cv2.LINE_AA,
            )

            # lane dash lines (left + right)
            vp_x = VP_X_BASE + int(
                VP_X_AMPLITUDE
                * math.sin(TWO * math.pi * frame_index / VP_OSC_PERIOD_FRAMES)
            )

            # left lane: (LEFT_LANE_BOTTOM_X, HEIGHT) -> (vp_x, VP_Y)
            left_x = np.linspace(LEFT_LANE_BOTTOM_X, float(vp_x), LANE_DASH_POINTS)
            left_y = np.linspace(float(LANE_BOTTOM_Y), float(VP_INT_Y_START), LANE_DASH_POINTS)
            for seg_i in range(0, LANE_DASH_POINTS - 1, DASH_SEGMENT_STEP):
                p0 = (int(left_x[seg_i]), int(left_y[seg_i]))
                p1 = (int(left_x[seg_i + 1]), int(left_y[seg_i + 1]))
                cv2.line(
                    frame,
                    p0,
                    p1,
                    LANE_DASH_COLOR,
                    thickness=LANE_DASH_SEGMENT_THICKNESS,
                    lineType=cv2.LINE_AA,
                )

            # right lane: (RIGHT_LANE_BOTTOM_X, HEIGHT) -> (vp_x, VP_Y)
            right_x = np.linspace(RIGHT_LANE_BOTTOM_X, float(vp_x), LANE_DASH_POINTS)
            right_y = np.linspace(float(LANE_BOTTOM_Y), float(VP_INT_Y_START), LANE_DASH_POINTS)
            for seg_i in range(0, LANE_DASH_POINTS - 1, DASH_SEGMENT_STEP):
                p0 = (int(right_x[seg_i]), int(right_y[seg_i]))
                p1 = (int(right_x[seg_i + 1]), int(right_y[seg_i + 1]))
                cv2.line(
                    frame,
                    p0,
                    p1,
                    LANE_DASH_COLOR,
                    thickness=LANE_DASH_SEGMENT_THICKNESS,
                    lineType=cv2.LINE_AA,
                )

            # moving blobs
            for blob in blobs:
                blob["x"] = float(blob["x"]) + float(blob["vx"])
                blob["y"] = float(blob["y"]) + float(blob["vy"])

                if blob["y"] > float(BLOB_OFFSCREEN_Y):
                    blob["y"] = float(BLOB_RESET_Y)
                    blob["x"] = float(
                        np.random.uniform(BLOB_X_RANDOM_MIN, BLOB_X_RANDOM_MAX)
                    )

                if blob["x"] < float(BLOB_X_BOUNCE_MIN) or blob["x"] > float(BLOB_X_BOUNCE_MAX):
                    blob["vx"] = -float(blob["vx"])

                x0 = int(blob["x"])
                y0 = int(blob["y"])
                x1 = x0 + int(blob["w"])
                y1 = y0 + int(blob["h"])
                cv2.rectangle(
                    frame, (x0, y0), (x1, y1), blob["color"], thickness=FILLED_THICKNESS
                )

            # brightness variation
            scale = ONE + BRIGHTNESS_WOBBLE_AMPLITUDE * math.sin(
                TWO * math.pi * frame_index / float(BRIGHTNESS_WOBBLE_PERIOD_FRAMES)
            )
            frame = (
                np.clip(frame.astype(np.float32) * scale, PIXEL_MIN, PIXEL_MAX).astype(np.uint8)
            )

            # gaussian noise
            noise = np.random.normal(NOISE_MEAN, NOISE_STD, frame.shape).astype(np.float32)
            frame = (
                np.clip(frame.astype(np.float32) + noise, PIXEL_MIN, PIXEL_MAX).astype(np.uint8)
            )

            out.write(frame)
    finally:
        out.release()

    print(
        "Sample video generated: samples/street.mp4 — "
        "300 frames, 640x360, 20fps"
    )
    print("Run: python main.py --source sample")


if __name__ == "__main__":
    main()

