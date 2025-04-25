from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams, PointStruct
from pymongo import MongoClient
from tqdm import tqdm
import time

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
    db = client["Normalized_db"]
    pbar.update(1)
    qdrant_client = QdrantClient(host="localhost", port=6333)
    pbar.update(1)

# Initialize collections with loading message
print("Preparing collections...")
collections = [
    db["final_target_1"],
    db["final_target_2"],
    db["final_target_3"]
]

# Load embedding model with progress indicator
print("Loading embedding model...")
with tqdm(total=1, desc="Loading model") as pbar:
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    pbar.update(1)

# Process all collections with detailed progress bars
points = []
current_id = 0
total_docs = sum(collection.count_documents({}) for collection in collections)
skipped=0
with tqdm(total=total_docs, desc="Processing documents") as pbar_docs:
    for collection in collections:
        collection_name = collection.name
        with tqdm(collection.find(), desc=f"Processing {collection_name}", leave=False) as pbar_col:
            for entry in pbar_col:
                #combined = " | ".join([n["name"] for n in entry["names"]])
                valid_names = [n["name"] for n in entry["names"] if is_valid_name(n.get("name"))]
                 # Skip this entry if no valid names found
                if not valid_names:
                    pbar_docs.update(1)
                    skipped+=1
                    pbar_col.update(1)
                    continue

                combined = " | ".join(valid_names)

                # Batch embedding for better performance
                with tqdm(total=1, desc="Embedding", leave=False) as pbar_emb:
                    names_embedding = encoder.encode(combined).tolist()
                    pbar_emb.update(1)
                
                points.append(
                    PointStruct(
                        id=current_id,
                        vector=names_embedding,
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
        collection_name="names_of_vectors_with_concatenation",
        vectors_config=VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=Distance.COSINE
        )
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
            collection_name="names_of_vectors_with_concatenation",
            points=batch
        )
        pbar.update(1)
        pbar.set_postfix({"Last ID": batch[-1].id if batch else 0})

print("\nOperation completed successfully!")
print(f"Total documents processed: {current_id}")
print(f"Total vectors stored: {len(points)}")
print(f"\nFinal stats: {current_id} valid points created, {total_docs - current_id} entries skipped")