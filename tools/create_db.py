import sqlite3

# List of wines with names as keys and URLs as values
wines = {
    "Chateau Margaux": "https://www.wine.com/product/chateau-margaux-2015/174984",
    "Penfolds Grange": "https://www.wine.com/product/penfolds-grange-2016/567112",
    "Opus One": "https://www.wine.com/product/opus-one-2017/549719",
    "Sassicaia": "https://www.wine.com/product/sassicaia-2018/233834",
    "Domaine de la Romanée-Conti La Tâche": "https://www.wine.com/product/domaine-de-la-romanee-conti-la-tache-2016/508206",
    "Chateau Latour": "https://www.wine.com/product/chateau-latour-2015/183926",
    "Mouton Rothschild": "https://www.wine.com/product/mouton-rothschild-2015/188124",
    "Harlan Estate": "https://www.wine.com/product/harlan-estate-2014/560338",
    "Chateau Haut-Brion": "https://www.wine.com/product/chateau-haut-brion-2015/181064",
    "Joseph Phelps Insignia": "https://www.wine.com/product/joseph-phelps-insignia-2016/579274",
    "Louis Roederer Cristal Brut": "https://www.wine.com/product/louis-roederer-cristal-brut-2012/123476",
    "Chateau Cheval Blanc": "https://www.wine.com/product/chateau-cheval-blanc-2015/178976",
    "Chateau d'Yquem": "https://www.wine.com/product/chateau-dyquem-2015/174964",
    "Chateau Lafite Rothschild": "https://www.wine.com/product/chateau-lafite-rothschild-2015/184864",
    "Caymus Special Selection Cabernet Sauvignon": "https://www.wine.com/product/caymus-special-selection-cabernet-sauvignon-2017/576089",
    "Screaming Eagle Cabernet Sauvignon": "https://www.wine.com/product/screaming-eagle-cabernet-sauvignon-2016/578467",
    "Tenuta San Guido Sassicaia": "https://www.wine.com/product/tenuta-san-guido-sassicaia-2016/560374",
    "Ornellaia Bolgheri Superiore": "https://www.wine.com/product/ornellaia-bolgheri-superiore-2017/577044",
    "E. Guigal Cote Rotie La Landonne": "https://www.wine.com/product/e-guigal-cote-rotie-la-landonne-2016/548027",
    "Vega Sicilia Unico": "https://www.wine.com/product/vega-sicilia-unico-2009/532019",
    "Chateau Palmer": "https://www.wine.com/product/chateau-palmer-2015/183826",
    "Petrus": "https://www.wine.com/product/petrus-2015/174964",
    "Chateau Mouton Rothschild": "https://www.wine.com/product/chateau-mouton-rothschild-2015/174984",
    "Chateau Margaux": "https://www.wine.com/product/chateau-margaux-2015/174984",
    "Chateau Haut-Brion": "https://www.wine.com/product/chateau-haut-brion-2015/181064",
    "Chateau Lafite Rothschild": "https://www.wine.com/product/chateau-lafite-rothschild-2015/184864",
    "Chateau Latour": "https://www.wine.com/product/chateau-latour-2015/183926",
    "Chateau Cheval Blanc": "https://www.wine.com/product/chateau-cheval-blanc-2015/178976",
    "Chateau d'Yquem": "https://www.wine.com/product/chateau-dyquem-2015/174964",
    "Caymus Special Selection Cabernet Sauvignon": "https://www.wine.com/product/caymus-special-selection-cabernet-sauvignon-2017/576089"
}
def create_wine_links_table():
    conn = sqlite3.connect('wine_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wine_links (
            wine TEXT PRIMARY KEY,
            link TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

create_wine_links_table()
def insert_wine_data(wines):
    conn = sqlite3.connect('wine_database.db')
    cursor = conn.cursor()
    
    # Inserting wine data into the database
    for wine, url in wines.items():
        cursor.execute('''
            INSERT OR REPLACE INTO wine_links (wine, link) VALUES (?, ?)
        ''', (wine, url))
    
    conn.commit()
    conn.close()

create_wine_links_table()
insert_wine_data(wines)
