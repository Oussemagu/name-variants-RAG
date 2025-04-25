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

# Load embedding model with progress indicator
print("Loading embedding model...")
with tqdm(total=1, desc="Loading model") as pbar:
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    pbar.update(1)

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

# Process all collections with detailed progress bars
points = []
current_id = 0
total_docs = sum(collection.count_documents({}) for collection in collections)
skipped=0
print("processing and uploading docs\n")
with tqdm(total=total_docs, desc="Processing documents") as pbar_docs:
    for collection in collections:
        collection_name = collection.name
        with tqdm(collection.find(), desc=f"Processing {collection_name}", leave=False) as pbar_col:
            for entry in pbar_col:
                #combined = " | ".join([n["name"] for n in entry["names"]])
                valid_names = [n["name"] for n in entry["names"] if is_valid_name(n.get("name"))]
                 # Skip this entry if no valid names found
                if not valid_names:
                    #print("this is invalid name",valid_names)
                    pbar_docs.update(1)
                    skipped+=1
                    pbar_col.update(1)
                    continue

                combined = " | ".join(valid_names)
                sparse_embedding = bm25.transform(combined)
                dense_embedding = encoder.encode(combined).tolist()
                # Batch embedding for better performance
               # with tqdm(total=1, desc="Embedding", leave=False) as pbar_emb:
                #    names_embedding = encoder.encode(combined).tolist()
                 #   pbar_emb.update(1)
                
                points.append(
                    PointStruct(
                        id=current_id,
                        #vector=names_embedding,
                        vector={
                            "text-sparse": models.SparseVector(indices=sparse_embedding["indices"],
                                                               values=sparse_embedding["values"]),
                            "text-dense":dense_embedding
                        },
                        #Qdrant stores  vector embeddings along with the optional JSON-like payload.
                        # While payloads are optional, LangChain assumes embeddings come from documents.
                        #By default, the document is going to be stored in the following payload structure "page_content" and "metadata"
                        # Keeping the payload allows retrieval of the original  text of embedding .
                        payload={
                            #"page_content": [n["name"] for n in entry["names"]],
                            "page_content":combined,  # String version of valid names
                            "metadata": {
                                "type": entry["type"],
                                "source": collection_name,
                                 "raw_names": valid_names  # Store original valid names
                            }
                        }
                    )
                )
                current_id += 1
                
                pbar_docs.update(1)
                pbar_col.set_postfix({"Processed": current_id,
                                    "Skipped": pbar_docs.n - current_id  # Show skipped count
                                      })

# Create collection with progress indicator
print("\nCreating Qdrant collection...")
with tqdm(total=1, desc="Creating collection") as pbar:
    qdrant_client.recreate_collection(
        #create a collection with sparse vector support 
        collection_name="names_of_vectors_with_concatenation_hybrid",
            ##vectors_config=VectorParams(
            ##size=encoder.get_sentence_embedding_dimension(),
            ##distance=Distance.COSINE
        vectors_config= {
            "text-dense":VectorParams(
                size=encoder.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
        "text-sparse": models.SparseVectorParams(
            index=models.SparseIndexParams(
                on_disk=False,
            )
        )
    },
    )
    pbar.update(1)

# Upload points in batches with progress
batch_size = 500
total_batches = (len(points) + batch_size - 1) // batch_size

print(f"\nUploading {len(points)} points in {total_batches} batches...")
with tqdm(total=total_batches, desc="Uploading points") as pbar:
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upload_points(
            collection_name="names_of_vectors_with_concatenation_hybrid",
            points=batch
        )
        pbar.update(1)
        pbar.set_postfix({"Last ID": batch[-1].id if batch else 0})

print("\nOperation completed successfully!")
print(f"Total documents processed: {current_id}")
print(f"Total vectors stored: {len(points)}")
print(f"\nFinal stats: {current_id} valid points created, {total_docs - current_id} entries skipped")
