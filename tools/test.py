from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")  # Adjust the URL to your Qdrant instance

tt = "Vega Sicilia Unico"

# Test the db
search_result = client.query(
    collection_name="wines4",
    query_text=tt
)
print(search_result)