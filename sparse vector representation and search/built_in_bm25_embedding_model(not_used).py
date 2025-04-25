from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams, PointStruct
from pymongo import MongoClient
from tqdm import tqdm
import time
#imports for BM25Tokenizer
from collections import defaultdict
import numpy as np
import hashlib
import math
#end
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding 
from qdrant_client import QdrantClient
from qdrant_client.models import  SearchRequest
bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
client = QdrantClient("localhost", port=6333)

def search_names(query_text, limit=5):
    return client.search(
        collection_name="names_of_vectors_with_concatenation_sparse",
        query=SearchRequest(
            query_text=query_text,
            query_embedding=bm25_embedding_model,
            limit=limit,
            with_payload=True
        )
    )

# Usage
results = search_names("mohammed abassi")
for hit in results:
    print(f"Score: {hit.score}")
    print(f"Match: {hit.payload['page_content']}")
    print(f"Type: {hit.payload['metadata']['type']}")