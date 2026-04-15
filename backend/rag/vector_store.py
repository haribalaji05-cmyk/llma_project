from typing import List

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class FaissVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.documents: List[str] = []
        self._embeddings = np.empty((0, dimension), dtype="float32")
        self.index = faiss.IndexFlatL2(dimension) if faiss is not None else None

    def add_documents(self, documents, embeddings) -> None:
        if len(embeddings) == 0:
            return

        vectors = np.array(embeddings, dtype="float32")
        self.documents.extend(documents)

        if self.index is not None:
            self.index.add(vectors)
            return

        self._embeddings = np.vstack([self._embeddings, vectors])

    def search(self, query_embedding, k=3):
        if not self.documents:
            return []

        query_vector = np.array(query_embedding, dtype="float32")
        if self.index is not None:
            _, indices = self.index.search(query_vector, k)
            return self._documents_from_indices(indices[0])

        scores = self._embeddings @ query_vector[0]
        top_indices = np.argsort(scores)[::-1][:k]
        return self._documents_from_indices(top_indices)

    def _documents_from_indices(self, indices):
        results = []
        for idx in indices:
            if 0 <= int(idx) < len(self.documents):
                results.append(self.documents[int(idx)])
        return results
