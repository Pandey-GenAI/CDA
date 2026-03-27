from qdrant_client import QdrantClient

# Initialize Qdrant client
client = QdrantClient("http://localhost:6333")  # Adjust the URL to your Qdrant instance

# Define wine descriptions
wines_with_descriptions = [
    "Chateau Margaux a prestigious Bordeaux wine with rich aromas of blackcurrant, tobacco, and licorice.",
    "Penfolds Grange an iconic Australian Shiraz known for its full-bodied structure and deep, complex flavors.",
    "Opus On a Napa Valley classic blending Cabernet Sauvignon and Merlot for a balanced, elegant taste.",
    "Sassicaia a renowned Italian wine from Tuscany, offering flavors of dark berries, spice, and a touch of oak.",
    "Domaine de la Romanée-Conti La Tâche an exceptional Burgundy wine with notes of red fruit, earth, and subtle floral hints.",
    "Chateau Latour a powerful and structured Bordeaux wine with flavors of cassis, cedar, and earthy undertones.",
    "Mouton Rothschild a celebrated Bordeaux wine known for its rich, velvety texture and complex layers of flavor.",
    "Harlan Estate a luxurious Napa Valley wine with deep, concentrated flavors of blackberry, chocolate, and espresso.",
    "Chateau Haut-Brion an elegant Bordeaux wine with notes of plum, tobacco, and a hint of minerality.",
    "Joseph Phelps Insignia a robust Napa Valley blend with flavors of dark fruit, cocoa, and a touch of vanilla.",
    "Louis Roederer Cristal Brut a prestigious Champagne with crisp acidity, fine bubbles, and notes of citrus and brioche.",
    "Chateau Cheval Blanc a refined Bordeaux wine with flavors of red fruit, spice, and a long, elegant finish.",
    "Chateau d'Yquem a legendary sweet wine from Sauternes, known for its2 honeyed flavors and luscious texture.",
    "Chateau Lafite Rothschild a premier Bordeaux wine with complex aromas of blackcurrant, graphite, and a hint of violets.",
    "Caymus Special Selection Cabernet Sauvignon a rich and opulent Napa Valley Cabernet with flavors of ripe berries and cocoa.",
    "Screaming Eagle Cabernet Sauvignon a rare and highly sought-after Napa Valley wine with intense flavors of dark fruit and spice.",
    "Tenuta San Guido Sassicaia a distinguished Italian wine from Bolgheri, offering notes of blackcurrant, herbs, and a touch of oak.",
    "Ornellaia Bolgheri Superiore a luxurious Italian wine with rich flavors of dark berries, tobacco, and chocolate.",
    "E. Guigal Cote Rotie La Landonne a powerful and complex Rhone wine with flavors of blackberry, smoked meat, and pepper.",
    "Vega Sicilia Unico a legendary Spanish wine with notes of dark fruit, spice, and a long, sophisticated finish.",
    "Chateau Palmer a refined Bordeaux wine with elegant aromas of plum, tobacco, and a hint of cedar.",
    "Petrus an exquisite Bordeaux wine from Pomerol, known for its rich, velvety texture and complex flavors.",
    "Chateau Mouton Rothschild a celebrated Bordeaux wine known for its rich, velvety texture and complex layers of flavor.",
    "Chateau Margaux a prestigious Bordeaux wine with rich aromas of blackcurrant, tobacco, and licorice.",
    "Chateau Haut-Brion an elegant Bordeaux wine with notes of plum, tobacco, and a hint of minerality.",
    "Chateau Lafite Rothschild a premier Bordeaux wine with complex aromas of blackcurrant, graphite, and a hint of violets.",
    "Chateau Latour a powerful and structured Bordeaux wine with flavors of cassis, cedar, and earthy undertones.",
    "Chateau Cheval Blanc a refined Bordeaux wine with flavors of red fruit, spice, and a long, elegant finish.",
    "Chateau d'Yquem a legendary sweet wine from Sauternes, known for its honeyed flavors and luscious texture.",
    "Caymus Special Selection Cabernet Sauvignon a rich and opulent Napa Valley Cabernet with flavors of ripe berries and cocoa."
]


# Use the new add method
client.add(
    collection_name="wines5",
    documents=wines_with_descriptions,
    ids=[i for i in range(0, len(wines_with_descriptions))]
)
# Test the db
search_result = client.query(
    collection_name="wines5",
    query_text="Chateau Cheval Blanc"
)
print(search_result)