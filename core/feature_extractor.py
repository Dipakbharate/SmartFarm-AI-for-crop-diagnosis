"""
core/feature_extractor.py

Singleton-style feature extractor for SmartFarm AI.

Responsibilities:
- Load the trained Keras CNN model once.
- Create a model that outputs the penultimate-layer representation (embedding).
- Provide convenient methods to extract embeddings from:
    - image file path (str / Path)
    - PIL.Image.Image
    - numpy array (HWC)
- Prefer using project's utils.image_utils preprocessing if available; otherwise use a robust fallback.

Usage:
    from core.feature_extractor import FeatureExtractor
    fe = FeatureExtractor.get_instance()
    embedding = fe.extract_from_path("data/uploads/sample_leaf.JPG")
    # embedding -> numpy.ndarray, shape (N,), dtype float32
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

# Lazy import tensorflow since it can be heavy
# Lazy import tensorflow since it can be heavy
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model, Model
except:
    tf = None
    load_model = None
    Model = None


# Try to import project preprocessing helper if available
try:
    from utils.image_utils import load_image, preprocess_image  # type: ignore
    _HAS_IMAGE_UTILS = True
except Exception:
    # fallback will be used
    _HAS_IMAGE_UTILS = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FeatureExtractor:
    """
    Feature extractor that loads the trained Keras model once and exposes methods to
    compute image embeddings from different input types.
    """

    def __init__(self,
                 model_path: Optional[Union[str, Path]] = None,
                 embedding_layer_index: Optional[int] = None):
        """
        :param model_path: Path to the .keras model file. If None, will resolve to ../model/smartfarm_cnn_best_model.keras.
        :param embedding_layer_index: If provided, will use model.layers[embedding_layer_index].output as embedding.
                                      Otherwise, the extractor will try to choose the penultimate layer.
        """
        if tf is None:
            raise RuntimeError("TensorFlow is required by FeatureExtractor but could not be imported.")

        # Resolve default path relative to project root
        if model_path is None:
            # file is at core/feature_extractor.py -> project root is one parent up
            default_model = Path(__file__).resolve().parents[1] / "model" / "smartfarm_cnn_best_model.keras"
            model_path = default_model

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Keras model not found at: {self.model_path}")

        self._base_model: Optional[tf.keras.Model] = None
        self._embedding_model: Optional[tf.keras.Model] = None
        self.embedding_layer_index = embedding_layer_index if embedding_layer_index is not None else -2  


        # Load model immediately
        self._load_models()

    def _load_models(self):
    
        import numpy as np
        from tensorflow.keras.models import load_model, Model

        logger.info(f"Loading CNN model for embedding extraction: {self.model_path}")
        self._base_model = load_model(self.model_path)

        # 🩹 PATCH: Handle Sequential models with no defined input
        try:
            input_shape = self._base_model.layers[0].input_shape[0]
            if input_shape[1] and input_shape[2]:
                h, w = input_shape[1], input_shape[2]
            else:
                h, w = 128, 128
            dummy_input = np.zeros((1, h, w, 3), dtype=np.float32)
            
        except Exception as e:
            logger.warning("⚠️ Model warm-up failed (non-fatal): %s", e)

        # Build embedding extractor
        try:
            embedding_output = self._base_model.layers[self.embedding_layer_index].output
            self._embedding_model = Model(inputs=self._base_model.input, outputs=embedding_output)
            logger.info(f"✅ Embedding model built using layer index {self.embedding_layer_index}.")
        except Exception as e:
            logger.warning(
              "⚠️ Failed to construct embedding model: %s. Falling back to base model outputs.", e
         )
    # fallback: use base model outputs as embedding (less ideal but safe)
            self._embedding_model = self._base_model


    def _ensure_loaded(self):
        if self._embedding_model is None:
            self._load_models()

    def _preprocess(self, img: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess PIL Image or numpy array into model-ready batch (shape: [1, H, W, C], dtype float32).
        Tries to use utils.image_utils preprocess/load helpers if present; else uses fallback logic.
        """
        # If the project has its own preprocessing helpers, use them
        if _HAS_IMAGE_UTILS:
            try:
                # Prefer a function named preprocess_image; otherwise load_image + preprocess_image
                if "preprocess_image" in globals() or hasattr(__import__("utils.image_utils", fromlist=["*"]), "preprocess_image"):
                    # import inside to avoid lint errors if not present
                    from utils.image_utils import preprocess_image as _pre  # type: ignore
                    arr = _pre(img) if callable(_pre) else None
                    if isinstance(arr, np.ndarray):
                        # ensure batch dim
                        if arr.ndim == 3:
                            arr = np.expand_dims(arr, axis=0)
                        return arr.astype(np.float32)
                # fallback: load_image then preprocess_image
                from utils.image_utils import load_image, preprocess_image  # type: ignore
                loaded = load_image(img) if callable(load_image) else img
                arr = preprocess_image(loaded)
                if arr.ndim == 3:
                    arr = np.expand_dims(arr, axis=0)
                return arr.astype(np.float32)
            except Exception as e:
                logger.warning("utils.image_utils preprocessing failed or not compatible: %s. Falling back to internal preprocessing.", e)

        # Fallback: internal preprocessing
        # Accept PIL.Image.Image or numpy array (H,W,C)
        if isinstance(img, np.ndarray):
            arr = img
            # if grayscale, convert to 3-channel
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
        else:
            if not isinstance(img, Image.Image):
                raise TypeError("img must be a PIL.Image.Image or numpy.ndarray")
            arr = np.array(img.convert("RGB"))

        # Determine target size from model.input_shape
        input_shape = None
        if self._embedding_model is not None:
            try:
                # model.input_shape -> (None, H, W, C) or similar
                ishape = self._embedding_model.input_shape
                if isinstance(ishape, (tuple, list)) and len(ishape) >= 3:
                    # find spatial dims
                    if ishape[1] is not None and ishape[2] is not None:
                        input_shape = (int(ishape[1]), int(ishape[2]))
            except Exception:
                input_shape = None

        if input_shape is None:
            # safe default
            input_shape = (128, 128)

        # Resize with PIL for good interpolation
        pil = Image.fromarray(arr.astype("uint8"), mode="RGB")
        pil = pil.resize(input_shape[::-1], Image.BILINEAR)  # PIL expects (width, height)
        arr = np.array(pil).astype("float32")

        # Normalization: scale to [0, 1]
        arr /= 255.0

        # Add batch dimension
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)

        return arr.astype(np.float32)

    def extract(self, img: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extract a 1-D embedding vector from an input.
        :param img: path-like, PIL.Image, or numpy array (HWC)
        :return: numpy.ndarray, shape (embedding_dim,), dtype float32
        """
        self._ensure_loaded()

        # Load PIL if path-like
        if isinstance(img, (str, Path)):
            img_path = Path(img)
            if not img_path.exists():
                raise FileNotFoundError(f"Image path does not exist: {img_path}")
            pil = Image.open(str(img_path)).convert("RGB")
            arr = self._preprocess(pil)
        else:
            arr = self._preprocess(img)  # handles PIL or numpy

        # Run through embedding model
        try:
            embeddings = self._embedding_model.predict(arr, verbose=0)
        except Exception as e:
            logger.exception("Failed to compute embedding: %s", e)
            raise

        # embeddings may be (1, d) or (1, h, w, c) depending on model; flatten to 1D vector
        emb = np.asarray(embeddings)
        emb = emb.squeeze()
        if emb.ndim > 1:
            emb = emb.reshape(-1)

        # Ensure dtype
        emb = emb.astype(np.float32)

        return emb

    # Convenience wrappers
    def extract_from_path(self, path: Union[str, Path]) -> np.ndarray:
        return self.extract(path)

    def extract_from_pil(self, image: Image.Image) -> np.ndarray:
        return self.extract(image)

    def extract_from_array(self, array: np.ndarray) -> np.ndarray:
        return self.extract(array)

    # Singleton helper
    @staticmethod
    @lru_cache(maxsize=1)
    def get_instance(model_path: Optional[Union[str, Path]] = None,
                     embedding_layer_index: Optional[int] = None) -> "FeatureExtractor":
        """
        Get a singleton FeatureExtractor for the process. Useful for Streamlit so the model is loaded once.
        Call without args to use the default model path found in the repository.
        If you pass model_path it will be used for the cached instance key.
        """
        return FeatureExtractor(model_path=model_path, embedding_layer_index=embedding_layer_index)


# Export
__all__ = ["FeatureExtractor"]
