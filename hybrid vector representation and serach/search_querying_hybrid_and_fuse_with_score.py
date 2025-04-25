from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams, PointStruct ,SearchRequest,NamedSparseVector
from pymongo import MongoClient
from tqdm import tqdm
import time
#imports for BM25Tokenizer
from collections import defaultdict
import numpy as np
import hashlib
import math
#end
print("Loading dense model...")
dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Loading dense model... done")
class BM25Tokenizer:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1 #BM25 param for term frequency scaling
        self.b=b #BM25 param for document length normalization
        self.doc_count=0
        self.avg_doc_length=0
        #document frequency for each term
        self.doc_freqs= defaultdict(int) #defaultdict(<class 'int'>,{:int})
        #defaultdict provide the default value for new keys
        #using itnb : default value will be 0
        self.term_to_index={} #mapping from terms to numerical indicies
        self.next_index= 1
    def tokenize(self,text):
        """Custom tokenizer optimized for names"""
        #lowercase , remove punctuatuation and split
        text= text.lower().replace("-"," ").replace("."," ").replace(",","")
        return [token for token in text.split() if token]
    def term_to_id(self,term):
        """ convert term to consistent numerical ID using hashing"""
        if term not in self.term_to_index:
            #use hash to create consistent numeric IDs
             hashed = int(hashlib.md5(term.encode()).hexdigest(), 16)
             self.term_to_index[term]= hashed % (2**31 - 1) #keep in positive int 32 range
        return self.term_to_index[term]  
    def fit(self,documents):
        """Calculate corpus statistics for BM25""" 
        total_length=0  
        self.doc_count=len(documents) #total number of documents in the corpus

        #forst pass : calculate document frequencies and average length
        for doc in documents:
            tokens = self.tokenize(doc) #word level tokenization
            total_length+=len(tokens) #total number of tokens in the document
            for term in set(tokens): #Only count once per document 
                self.doc_freqs[term]+=1
        self.avg_doc_length= total_length/self.doc_count if self.doc_count>0 else 0
        
    def transform(self,text):
        """Convert text to BM25 sparse vector"""
        tokens= self.tokenize(text) #word level tokenization
        term_counts = defaultdict(int)
        doc_length= len(tokens)

        #count TF in this document
        for term in tokens:
            term_counts[term]+=1
        
        #Calculate BM25 wieghts
        vector_indices= []
        vector_values = []
        for term, tf in term_counts.items():
            #skip terms not seen during filtering
            if term not in self.doc_freqs:
                continue

            # Calculate BM25 component
            idf = math.log((self.doc_count - self.doc_freqs[term] + 0.5) / 
                          (self.doc_freqs[term] + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            weight = idf * numerator / denominator
            
            vector_indices.append(self.term_to_id(term))
            vector_values.append(float(weight))
        return {
            "indices": vector_indices,
            "values" : vector_values
        }

def concatenate_names(entry):
    """Helper function to combine all names in an entry"""
    return " | ".join(n["name"] for n in entry.get("names", []))

def is_valid_name(name):
    """Check if a name is non-empty and contains non-whitespace characters"""
    return name and isinstance(name, str) and name.strip()

# Initialize connections with progress tracking
print("Initializing connections...")
with tqdm(total=3, desc="Setting up clients") as pbar:
    client = MongoClient("mongodb://localhost:27017/")
    pbar.update(1)
    db = client["extracted_names"]
    pbar.update(1)
    qdrant_client = QdrantClient(host="localhost", port=6333)
    pbar.update(1)

# Initialize collections with loading message
print("Preparing collections...")
collections = [
    db["extracted_names"]
]
# First collect all documents to train BM25
print("Collecting documents for BM25 training...")
all_documents = []
with tqdm(total=collections[0].count_documents({}), desc="Collecting documents") as pbar:
    for entry in collections[0].find():
        valid_names = [n["name"] for n in entry.get("names", []) if is_valid_name(n.get("name"))]
        if valid_names:
            all_documents.append(" | ".join(valid_names))
        pbar.update(1)

# Initialize and train BM25
print("Training BM25 tokenizer...")
bm25 = BM25Tokenizer()
bm25.fit(all_documents)

def hybrid_name_search_1(query_text, limit=10):
    # Generate both vector types
    dense_vector = dense_model.encode(query_text).tolist()
    sparse_vector = bm25.transform(query_text)
    
    # Create search requests
    dense_query = models.SearchRequest(
        vector=models.NamedVector(
            name="text-dense",
            vector=dense_vector
        ),
        limit=20,
        with_payload=True
    )
    
    sparse_query = models.SearchRequest(
        vector=models.NamedSparseVector(
            name="text-sparse",
            vector=models.SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"]
            )
        ),
        limit=20,
        with_payload=True
    )
    
    # Execute the search with RRF fusion
    response = qdrant_client.search_batch(
        collection_name="names_of_vectors_with_concatenation_hybrid",
        requests=[dense_query, sparse_query],
        #fusion=models.Fusion.RRF,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit
    )
    
    # Process results
    return [{
        "score": hit.score,
        "text": hit.payload["page_content"],
        "type": hit.payload["metadata"]["type"],
        "raw_names": hit.payload["metadata"]["raw_names"]
    } for hit in response]

#parallel retrieval of dense and sparse results  using Prefetch
#before fusion(aka mixing): normalization of score / score fusion

def hybrid_name_search(query_text, limit=10):
    # Generate both vector types
    dense_vector = dense_model.encode(query_text).tolist()
    sparse_vector = bm25.transform(query_text)
    
    # Create prefetch requests
    prefetch_queries = [
        models.Prefetch(
            query=dense_vector,
            using="text-dense",  # Must match vectors_config key
            limit=20  # Get more candidates than final limit
        ),
        models.Prefetch(
            query=models.SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"]
            ),
            using="text-sparse",
            limit=20
        )
    ]
    
    # Execute parallel search
   # Execute search 
    results = qdrant_client.query_points(
        collection_name="names_of_vectors_with_concatenation_hybrid",
        prefetch=prefetch_queries,
        limit=limit,
        #fusion=models.Fusion.RRF
        #Fusion method Reciprocal Rank Fusion (RRF)
        query=models.FusionQuery(fusion=models.Fusion.DBSF),
        )
    
   
        
    return results.points


def print_search_results(results):
    print(f"\nFound {len(results)} matches:")
    for i, point in enumerate(results, 1):
        print(f"\nMatch #{i}:")
        print(f"ID: {point.id}")
        print(f"Score: {point.score:.4f}")
        print(f"Content: {point.payload['page_content']}")
        print(f"Type: {point.payload['metadata']['type']}")
        print("Original Names:")
        for name in point.payload['metadata']['raw_names']:
            print(f"  - {name}")
        print(f"Source: {point.payload['metadata']['source']}")
results = hybrid_name_search("أسامة قريش", limit=3)
print_search_results(results)

