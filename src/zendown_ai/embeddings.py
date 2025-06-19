# zendown_ai/embeddings.py

from lancedb.embeddings.registry import register
from lancedb.embeddings import TextEmbeddingFunction
from lancedb.util import attempt_import_or_raise
from functools import lru_cache

sentence_transformers = attempt_import_or_raise("sentence_transformers")


@register("sentence-transformers")
class SentenceTransformerEmbeddings(TextEmbeddingFunction):
    name: str = "all-MiniLM-L6-v2"

    def __init__(self, name: str = None, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self._name = name or self.name
        self._device = device
        self._ndims = None
        self._model = self._embedding_model()

    def generate_embeddings(self, texts):
        if not texts:
            return []
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self._model.encode(
            texts,
            convert_to_tensor=False,
            device=self._device
        )
        return [emb.tolist() for emb in embeddings]

    def ndims(self):
        if self._ndims is None:
            dummy_embedding = self.generate_embeddings("foo")
            if dummy_embedding and len(dummy_embedding[0]) > 0:
                self._ndims = len(dummy_embedding[0])
            else:
                try:
                    self._ndims = self._model.get_sentence_embedding_dimension()
                except Exception as e:
                    print(f"Warning: Could not determine embedding dimension: {
                          e}. Defaulting to 384.")
                    self._ndims = 384
        return self._ndims

    @lru_cache(maxsize=1)
    def _embedding_model(self):
        return sentence_transformers.SentenceTransformer(self._name, device=self._device)

    def source_columns(self):
        return ["text"]

    def model_name(self):
        return self._name
