import os
import json
import logging
import numpy as np
import cv2
import threading
from typing import Dict, List, Optional, Tuple

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
        
        # Cache: { (dataset_version, reference_id): embedding_vector }
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

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Resize to input_size (squashing), Normalize, CHW, Batch dim.
        Expected input: BGR image (H, W, 3)
        """
        # Resize (squash)
        # cv2.resize expects (width, height)
        resized = cv2.resize(image, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and standardized if needed
        # MobileNet expects specific mean/std usually, but let's stick to simple [0,1] or standard
        # if the training script used standard ImageNet normalization.
        # For this implementation, let's assume standard ImageNet normalization
        # as we used `models.mobilenet_v2(weights=...)` in the training script 
        # but didn't specify transforms.
        # Let's use standard ImageNet mean/std for safety as that's what the model expects.
        img_data = rgb.astype(np.float32) / 255.0
        
        # ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        
        # HWC -> CHW
        img_data = img_data.transpose(2, 0, 1)
        
        # Add batch dimension -> BCHW
        img_data = np.expand_dims(img_data, axis=0)
        
        return img_data

    def compute_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.session is None:
            return None

        try:
            input_tensor = self.preprocess(image)
            input_name = self.session.get_inputs()[0].name
            
            # Run inference
            outputs = self.session.run(None, {input_name: input_tensor})
            embedding = outputs[0][0] # First batch element
            
            return embedding
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None

    def get_reference_embedding(self, dataset_version: str, ref_id: str, ref_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns cached embedding or computes and caches it.
        """
        key = (dataset_version, ref_id)
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
        # Vectors from the model (via torch.nn.functional.normalize) should already be unit vectors if 
        # we used that layer.
        # But let's be safe.
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0 # Max distance
            
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        # Clamp for numerical stability
        similarity = np.clip(similarity, -1.0, 1.0)
        
        return 1.0 - similarity
