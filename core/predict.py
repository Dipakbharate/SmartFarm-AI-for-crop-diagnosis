"""
core/predict.py
-----------------------------------------
Performs disease prediction using the trained SmartFarm CNN model.

Responsibilities:
- Load Keras model once (cached).
- Preprocess input image using utils.image_utils if available.
- Predict disease class and confidence.
- Provide helper functions for Streamlit / CLI use.

Example:
    from core.predict import predict_image
    label, conf, topk = predict_image("data/uploads/sample_leaf.JPG")
    print(label, conf)
"""

import json
import logging
from pathlib import Path
from functools import lru_cache
from typing import Union, Tuple, List

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Try to use project utils
try:
    from utils.image_utils import load_image, preprocess_image  # type: ignore
    _HAS_IMAGE_UTILS = True
except Exception:
    _HAS_IMAGE_UTILS = False


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -------------------- Configuration --------------------
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "smartfarm_cnn_best_model.keras"

CLASS_INDICES_PATH = Path(__file__).resolve().parents[1] / "model" / "class_indices.json"


# -------------------- Model Loader --------------------
@lru_cache(maxsize=1)
def load_cnn_model():
    """Load and cache the CNN model (once per session)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    logger.info("Loading SmartFarm CNN model from %s", MODEL_PATH)
    model = load_model(str(MODEL_PATH), compile=False)
    logger.info("Model loaded successfully.")
    return model


@lru_cache(maxsize=1)
def load_class_indices() -> dict:
    """Load class index mapping."""
    if not CLASS_INDICES_PATH.exists():
        raise FileNotFoundError(f"Class indices JSON not found: {CLASS_INDICES_PATH}")
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        indices = json.load(f)
    # reverse mapping: idx -> class name
    index_to_class = {v: k for k, v in indices.items()}
    return index_to_class


# -------------------- Preprocessing --------------------
def _preprocess_image(img: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Preprocess an input image into a batch ready for CNN model.
    """
    if _HAS_IMAGE_UTILS:
        try:
            from utils.image_utils import preprocess_image  # type: ignore
            return preprocess_image(img)
        except Exception as e:
            logger.warning("utils.image_utils.preprocess_image failed (%s). Using internal fallback.", e)

    # Fallback preprocessing
    if isinstance(img, (str, Path)):
        img = Image.open(str(img)).convert("RGB")
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype("uint8")).convert("RGB")
    elif not isinstance(img, Image.Image):
        raise TypeError("Unsupported image type for preprocessing.")

    # Resize to model input shape if available
    model = load_cnn_model()
    ishape = model.input_shape
    if isinstance(ishape, (tuple, list)) and len(ishape) >= 3:
        target_size = (ishape[1], ishape[2])
    else:
        target_size = (128, 128)
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# -------------------- Prediction --------------------
def predict_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    top_k: int = 3
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Predict plant disease from an input image.

    Returns:
        predicted_label (str)
        confidence (float)
        topk (List[(label, probability)])
    """
    model = load_cnn_model()
    index_to_class = load_class_indices()

    # preprocess
    img_batch = _preprocess_image(image_input)

    # inference
    preds = model.predict(img_batch, verbose=0)
    preds = np.squeeze(preds)

    # probabilities and top-k
    top_indices = preds.argsort()[::-1][:top_k]
    top_labels = [(index_to_class[i], float(preds[i])) for i in top_indices]
    best_label, best_conf = top_labels[0]

    logger.info("Predicted: %s (%.2f%%)", best_label, best_conf * 100)
    return best_label, best_conf, top_labels


# -------------------- CLI Helper --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SmartFarm AI Disease Prediction")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to leaf image")
    parser.add_argument("--topk", "-k", type=int, default=3, help="Top-K predictions to show")
    args = parser.parse_args()

    label, conf, topk = predict_image(args.image, top_k=args.topk)
    print("\n✅ Prediction complete!")
    print(f"🌿 Predicted: {label}")
    print(f"📊 Confidence: {conf*100:.2f}%")
    print("\nTop-{0} predictions:".format(args.topk))
    for lbl, prob in topk:
        print(f" - {lbl:40s}: {prob*100:.2f}%")
