# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# from flask import Flask, request, jsonify

# # Load dataset
# df = pd.read_csv("data.csv")
# documents = df['document'].tolist()

# # Load pre-trained model
# model = SentenceTransformer('all-MiniLM-L6-v2')  # Small and efficient model

# # Generate embeddings for the documents
# print("Generating embeddings for documents...")
# embeddings = model.encode(documents, show_progress_bar=True)

# # Convert embeddings to numpy array
# embeddings_np = np.array(embeddings)

# # Create FAISS index
# dimension = embeddings_np.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings_np)
# print(f"Number of documents in the index: {index.ntotal}")

# # Save the FAISS index
# faiss.write_index(index, "document_index.faiss")

# # Flask app
# app = Flask(__name__)

# @app.route("/query", methods=["POST"])
# def search():
#     data = request.json
#     query = data["query"]

#     # Generate embedding for query
#     query_embedding = model.encode([query])

#     # Search for the closest match (replace top_k with k)
#     k = 1  # We want the top 1 result
#     distances, indices = index.search(query_embedding, k)  # Changed top_k to k
    
#     # Fetch the best match document
#     best_match = documents[indices[0][0]]  # indices is a 2D array, so we access the first element
#     distance = float(distances[0][0])  # Convert the numpy float32 to a Python float

#     return jsonify({"best_match": best_match, "distance": distance})

# if __name__ == "__main__":
#     app.run(debug=True)


# following is 2nd code snippet it does not contain any  model


 # import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# from flask import Flask, request, jsonify
# from flask_cors import CORS  # Import CORS handling
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# # Load dataset
# df = pd.read_csv("data.csv")
# documents = df['document'].tolist()

# # Load pre-trained model for embeddings
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Small and efficient

# # Generate embeddings for the documents
# print("Generating embeddings for documents...")
# embeddings = embedding_model.encode(documents, show_progress_bar=True)

# # Convert embeddings to numpy array
# embeddings_np = np.array(embeddings)

# # Create FAISS index
# dimension = embeddings_np.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings_np)
# print(f"Number of documents in the index: {index.ntotal}")

# # Save the FAISS index
# faiss.write_index(index, "document_index.faiss")

# # Load an open-source LLM (Llama 2 or Hugging Face model)
# print("Loading LLM model...")
# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")  # Small LLM for demo
# llm_model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
# llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)

# # Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# @app.route("/query", methods=["POST"])
# def search_and_answer():
#     data = request.json
#     query = data["query"]

#     # 1. Generate embedding for the query
#     query_embedding = embedding_model.encode([query])

#     # 2. Search for the closest match
#     k = 1  # Top 1 result
#     distances, indices = index.search(query_embedding, k)
#     best_match = documents[indices[0][0]]  # Get the document
#     distance = float(distances[0][0])  # Convert distance to a Python float

#     # 3. Pass the document and query to the LLM
#     prompt = (
#         f"Document: {best_match}\n\n"
#         f"Question: {query}\n\n"
#         f"Answer:"
#     )
#     response = llm_pipeline(prompt, max_length=1000, num_return_sequences=1)
#     answer = response[0]['generated_text'].split("Answer:")[-1].strip()

#     # 4. Return the result
#     return jsonify({
#         "best_match": best_match,
#         "distance": distance,
#         "answer": answer
#     })

# if __name__ == "__main__":
#     app.run(debug=True)


  #following is the 3rd code snippet it contains bigscience/bloomz-560m model


import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS handling
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load dataset
df = pd.read_csv("data.csv")
documents = df['document'].tolist()

# Load pre-trained model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Small and efficient

# Generate embeddings for the documents
print("Generating embeddings for documents...")
embeddings = embedding_model.encode(documents, show_progress_bar=True)

# Convert embeddings to numpy array
embeddings_np = np.array(embeddings)

# Create FAISS index
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)
print(f"Number of documents in the index: {index.ntotal}")

# Save the FAISS index
faiss.write_index(index, "document_index.faiss")

# Load an open-source LLM (Llama 2 or Hugging Face model)
print("Loading LLM model...")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")  # Small LLM for demo
llm_model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)

# Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/query", methods=["POST"])
def search_and_answer():
    try:
        data = request.json
        query = data["query"]

        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400

        # 1. Generate embedding for the query
        query_embedding = embedding_model.encode([query])

        # 2. Search for the top-k closest matches
        k = 3  # Top 3 results for more accurate responses
        distances, indices = index.search(query_embedding, k)
        
        # Fetch the best matches
        best_matches = [documents[i] for i in indices[0]]
        distances = [float(dist) for dist in distances[0]]  # Convert distances to floats

        # 3. Pass the documents and query to the LLM
        prompt = (
            f"Documents:\n{best_matches}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        response = llm_pipeline(prompt, max_length=1500, num_return_sequences=1,temperature=0.9)
        answer = response[0]['generated_text'].split("Answer:")[-1].strip()

        # 4. Return the result
        return jsonify({
            "best_matches": best_matches,
            "distances": distances,
            "answer": answer
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

