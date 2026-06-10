"""
faiss_index.py
--------------
Build, query, and serialize the FAISS item index.

The FAISS IVF (Inverted File Index) partitions the item embedding space
into k-means Voronoi regions. At query time, only nprobe nearest regions
are searched — giving ~10x speedup over exact search with <1% recall loss.

Architecture fit:
  - The index stores item tower embeddings ONLY (pre-computed offline)
  - At query time, the combined user+query vector searches the index
  - The index never changes at query time — only rebuilt after retraining

Usage:
    # Build index from trained model
    python scripts/build_faiss_index.py --config config2

    # Query the index
    from retrieval.faiss_index import FaissIndex
    index = FaissIndex.load("artifacts/faiss_index/config2.bin")
    distances, indices = index.search(query_vectors, k=100)
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class FaissIndex:
    """
    Wrapper around FAISS IVF index for item retrieval.

    Handles:
      - Building from item embeddings
      - Serialization / deserialization
      - ANN search with configurable nprobe
      - Index statistics logging
    """

    def __init__(
        self,
        embed_dim:  int,
        index_type: str = "ivf",   # "flat" or "ivf"
        nlist:      int = 100,     # IVF: number of Voronoi cells
        nprobe:     int = 10,      # IVF: cells to search at query time
    ):
        self.embed_dim  = embed_dim
        self.index_type = index_type
        self.nlist      = nlist
        self.nprobe     = nprobe
        self.index      = None
        self.item_ids   = None     # maps index position → asin

        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. Run: pip install faiss-cpu"
            )

    def build(
        self,
        item_embeddings: np.ndarray,
        item_ids:        np.ndarray,
    ) -> None:
        """
        Build the FAISS index from item embeddings.

        Args:
            item_embeddings: [num_items, embed_dim] — L2-normalized item vectors
            item_ids:        [num_items] — corresponding item identifiers (asins)
        """
        assert item_embeddings.shape[1] == self.embed_dim, (
            f"Embedding dim mismatch: got {item_embeddings.shape[1]}, "
            f"expected {self.embed_dim}"
        )

        n_items = len(item_embeddings)
        embeddings = item_embeddings.astype(np.float32)
        self.item_ids = item_ids

        logger.info(
            f"Building FAISS {self.index_type.upper()} index: "
            f"{n_items:,} items, embed_dim={self.embed_dim}"
        )

        if self.index_type == "flat":
            # Exact search — no approximation
            # Use inner product (equivalent to cosine sim on L2-normalized vectors)
            self.index = self.faiss.IndexFlatIP(self.embed_dim)
            self.index.add(embeddings)
            logger.info(f"  Flat index built: {self.index.ntotal:,} vectors")

        elif self.index_type == "ivf":
            # IVF: k-means partitioning into nlist Voronoi regions
            # Requires training on a representative sample first
            quantizer = self.faiss.IndexFlatIP(self.embed_dim)
            self.index = self.faiss.IndexIVFFlat(
                quantizer, self.embed_dim, self.nlist,
                self.faiss.METRIC_INNER_PRODUCT
            )

            # Train on all data (or a sample if very large)
            train_data = embeddings
            if n_items > 500_000:
                # Sample for training — index all items
                sample_idx = np.random.choice(n_items, 500_000, replace=False)
                train_data = embeddings[sample_idx]

            logger.info(f"  Training IVF quantizer on {len(train_data):,} samples...")
            self.index.train(train_data)
            self.index.add(embeddings)
            self.index.nprobe = self.nprobe

            logger.info(
                f"  IVF index built: {self.index.ntotal:,} vectors | "
                f"nlist={self.nlist} | nprobe={self.nprobe}"
            )

        else:
            raise ValueError(f"Unknown index_type: {self.index_type}. Use 'flat' or 'ivf'.")

    def search(
        self,
        query_vectors: np.ndarray,
        k:             int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for the k nearest items to each query vector.

        Args:
            query_vectors: [batch_size, embed_dim] — L2-normalized query vectors
            k:             Number of nearest neighbors to retrieve

        Returns:
            distances: [batch_size, k] — similarity scores (higher = more similar)
            indices:   [batch_size, k] — positions in the index (map via item_ids)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        queries = query_vectors.astype(np.float32)
        distances, indices = self.index.search(queries, k)
        return distances, indices

    def get_item_ids(self, indices: np.ndarray) -> np.ndarray:
        """
        Convert FAISS index positions to item identifiers (ASINs).

        Args:
            indices: [batch_size, k] — FAISS index positions

        Returns:
            [batch_size, k] — ASINs
        """
        if self.item_ids is None:
            raise RuntimeError("item_ids not set. Build the index first.")
        return self.item_ids[indices]

    def set_nprobe(self, nprobe: int) -> None:
        """Adjust nprobe at runtime. Higher = slower but better recall."""
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe
            self.nprobe = nprobe
            logger.info(f"  nprobe set to {nprobe}")

    def save(self, path: str) -> None:
        """Serialize index and item_ids to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        self.faiss.write_index(self.index, str(path))

        # Save item_ids separately (pickle alongside index)
        ids_path = path.with_suffix(".ids.pkl")
        with open(ids_path, "wb") as f:
            pickle.dump({
                "item_ids":   self.item_ids,
                "embed_dim":  self.embed_dim,
                "index_type": self.index_type,
                "nlist":      self.nlist,
                "nprobe":     self.nprobe,
            }, f)

        size_mb = path.stat().st_size / 1024 / 1024
        logger.info(f"  Index saved: {path} ({size_mb:.1f} MB)")
        logger.info(f"  Item IDs saved: {ids_path}")

    @classmethod
    def load(cls, path: str) -> "FaissIndex":
        """Load a serialized index from disk."""
        path = Path(path)
        ids_path = path.with_suffix(".ids.pkl")

        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        if not ids_path.exists():
            raise FileNotFoundError(f"Item IDs not found: {ids_path}")

        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

        with open(ids_path, "rb") as f:
            meta = pickle.load(f)

        instance = cls(
            embed_dim  = meta["embed_dim"],
            index_type = meta["index_type"],
            nlist      = meta["nlist"],
            nprobe     = meta["nprobe"],
        )
        instance.index    = faiss.read_index(str(path))
        instance.item_ids = meta["item_ids"]

        if hasattr(instance.index, "nprobe"):
            instance.index.nprobe = meta["nprobe"]

        logger.info(
            f"Index loaded: {path} | "
            f"{instance.index.ntotal:,} vectors | "
            f"type={meta['index_type']} | "
            f"nprobe={meta['nprobe']}"
        )
        return instance

    def log_stats(self) -> None:
        """Log index statistics."""
        if self.index is None:
            logger.warning("Index not built yet.")
            return
        logger.info(
            f"FAISS index stats:\n"
            f"  Type:      {self.index_type.upper()}\n"
            f"  Vectors:   {self.index.ntotal:,}\n"
            f"  Embed dim: {self.embed_dim}\n"
            f"  nlist:     {self.nlist}\n"
            f"  nprobe:    {self.nprobe}\n"
            f"  Items:     {len(self.item_ids):,}"
        )
