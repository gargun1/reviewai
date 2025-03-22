from flask import Flask, request, jsonify
from azure.cosmos import CosmosClient
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Azure OpenAI Configuration
openai.api_key = "<YOUR_AZURE_OPENAI_KEY>"
openai.api_base = "<YOUR_AZURE_OPENAI_ENDPOINT>"
openai.api_type = "azure"
openai.api_version = "2023-05-15"

# Azure Cosmos DB Configuration
cosmos_client = CosmosClient("<YOUR_COSMOS_DB_CONNECTION_STRING>")
database = cosmos_client.get_database_client("<YOUR_DATABASE_NAME>")
container = database.get_container_client("<YOUR_CONTAINER_NAME>")

# Dummy Product Data (Replace with actual data from Cosmos DB)
products = [
    {"id": 1, "name": "Laptop", "description": "High-performance laptop with 16GB RAM."},
    {"id": 2, "name": "Smartphone", "description": "5G smartphone with 128GB storage."},
    {"id": 3, "name": "Tablet", "description": "10-inch tablet with long battery life."},
]

# NLP Function
def get_user_intent(query):
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",  # Replace with your deployed model name
        messages=[{"role": "user", "content": query}],
        max_tokens=50
    )
    return response['choices'][0]['message']['content']

# Recommendation Function
def recommend_products(query):
    vectorizer = TfidfVectorizer()
    product_descriptions = [p["description"] for p in products]
    tfidf_matrix = vectorizer.fit_transform(product_descriptions)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    recommended_indices = similarities.argsort()[-3:][::-1]
    return [products[i] for i in recommended_indices]

# API Endpoint
@app.route("/recommend", methods=["POST"])
def recommend():
    user_query = request.json.get("query")
    intent = get_user_intent(user_query)
    recommendations = recommend_products(intent)
    return jsonify({"intent": intent, "recommendations": recommendations})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
