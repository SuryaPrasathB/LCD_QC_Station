import os
import json
import logging
import numpy as np
import cv2
import threading
from typing import Dict, List, Optional, Tuple, Union

# Attempt to import onnxruntime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Singleton-like class to handle ONNX model loading and inference.
    Maintains a cache of reference embeddings.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.session = None
        self.meta = {}
        self.input_size = (128, 128)
        self.embedding_dim = 128
        
        # Cache: { (dataset_version, reference_id_hash): embedding_vector }
        # ref_id should be unique combination of file + roi coords
        self._reference_cache: Dict[Tuple[str, str], np.ndarray] = {}
        
        self.load_model()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_model(self):
        if not HAS_ONNX:
            logger.error("onnxruntime not installed. Cannot load embedding model.")
            return

        onnx_path = os.path.join(self.model_dir, "embedding_v2.onnx")
        meta_path = os.path.join(self.model_dir, "model_meta.json")

        if not os.path.exists(onnx_path):
            logger.error(f"Model not found at {onnx_path}")
            return

        try:
            # Load Metadata
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    self.meta = json.load(f)
                    self.input_size = tuple(self.meta.get("input_size", [128, 128]))
                    self.embedding_dim = self.meta.get("embedding_dim", 128)
            
            # Load ONNX Session
            # providers=['CPUExecutionProvider'] ensures CPU only as required
            self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            logger.info(f"Loaded embedding model from {onnx_path}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.session = None

    def preprocess(self, images: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Resize to input_size, Normalize, CHW, Batch dim.
        Accepts single image (H,W,3) or list of images.
        Returns tensor (B, 3, H, W).
        """
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]

        batch_data = []
        
        # ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for img in images:
            # Resize (squash)
            resized = cv2.resize(img, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR)

            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1]
            img_data = rgb.astype(np.float32) / 255.0

            # Standardize
            img_data = (img_data - mean) / std

            # HWC -> CHW
            img_data = img_data.transpose(2, 0, 1)

            batch_data.append(img_data)

        return np.array(batch_data, dtype=np.float32)

    def compute_embedding(self, images: Union[np.ndarray, List[np.ndarray]]) -> Optional[np.ndarray]:
        """
        Computes embedding for one or multiple images.
        Handles fixed batch-size models by iterating manually.
        Returns (B, 128) array or None on failure.
        """
        if self.session is None:
            return None

        # Helper to run single item
        def _run_single(img_tensor_1chw):
            input_name = self.session.get_inputs()[0].name
            # Expand dims handled in preprocess for single item?
            # preprocess returns (1, 3, 128, 128) for single item in list? Yes.
            outputs = self.session.run(None, {input_name: img_tensor_1chw})
            return outputs[0][0] # (128,)

        try:
            input_tensor = self.preprocess(images) # (B, 3, 128, 128)

            B = input_tensor.shape[0]
            embeddings = []
            
            for i in range(B):
                # Slice (1, 3, 128, 128)
                item = input_tensor[i:i+1]
                emb = _run_single(item)
                embeddings.append(emb)

            embeddings = np.array(embeddings) # (B, 128)
            
            # Compatibility return for single raw image input (H, W, 3)
            if isinstance(images, np.ndarray) and images.ndim == 3:
                return embeddings[0]

            return embeddings
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None

    def get_reference_embedding(self, dataset_version: str, ref_key: str, ref_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns cached embedding or computes and caches it.
        ref_key must be unique (include ROI hash if needed).
        """
        key = (dataset_version, ref_key)
        if key in self._reference_cache:
            return self._reference_cache[key]
        
        embedding = self.compute_embedding(ref_image)
        if embedding is not None:
            self._reference_cache[key] = embedding
            
        return embedding

    def compute_cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Computes cosine distance between two vectors.
        Distance = 1 - CosineSimilarity
        Vectors are assumed to be 1D.
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0 # Max distance
            
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        similarity = np.clip(similarity, -1.0, 1.0)
        
        return 1.0 - similarity

    def compute_batch_distances(self, candidate_embeddings: np.ndarray, ref_embedding: np.ndarray) -> np.ndarray:
        """
        Computes distances between a batch of candidates (B, D) and a single reference (D,).
        Returns (B,) array of distances.
        """
        # Normalize candidates
        norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) # (B, 1)
        candidates_norm = candidate_embeddings / (norms + 1e-8)

        # Normalize reference
        ref_norm = ref_embedding / (np.linalg.norm(ref_embedding) + 1e-8)

        # Dot product
        # (B, D) . (D,) -> (B,)
        similarities = np.dot(candidates_norm, ref_norm)

        return 1.0 - similarities
