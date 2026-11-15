"""Find people in a frame (MediaPipe selfie segmentation), so the app ignores whatever they cover.

    finder = PersonFinder()
    mask = finder.mask(frame)          # uint8, 255 where a person (or their hair/arm) is

Why: anything a person covers is not board writing. Without this, a region with someone's
hair and a chair was read as a made-up paragraph. The same mask will later let overlays be
drawn behind the professor.

The model file (~250 KB, MediaPipe's official "selfie segmenter") downloads once into models/.
"""
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_segmenter/float16/latest/selfie_segmenter.tflite")
MODEL_PATH = Path(__file__).resolve().parent / "models" / "selfie_segmenter.tflite"


class PersonFinder:
    def __init__(self, threshold=0.5, grow=0.02):
        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(exist_ok=True)
            print(f"Downloading the person-segmentation model to {MODEL_PATH} ...", flush=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        options = ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                                        output_confidence_masks=True)
        self.segmenter = ImageSegmenter.create_from_options(options)
        self.threshold = threshold
        self.grow = grow          # enlarge the mask by 2% of the frame: hair and arm edges are soft

    def close(self):
        """Call before the program ends (MediaPipe errors if it is cleaned up during Python's exit)."""
        if self.segmenter is not None:
            try:
                self.segmenter.close()
            except Exception:
                pass
            self.segmenter = None

    def mask(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (256, max(1, int(256 * h / w))))   # the model works at 256 px
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        confidence = self.segmenter.segment(image).confidence_masks[0].numpy_view()
        person = (cv2.resize(confidence, (w, h)) > self.threshold).astype(np.uint8) * 255
        k = max(3, int(min(h, w) * self.grow) | 1)
        return cv2.dilate(person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
