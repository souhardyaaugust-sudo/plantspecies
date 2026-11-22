import json

with open("advanced_plant_database.json", "r", encoding="utf-8") as f:
    db = json.load(f)

trees_data = {
    "Banyan Tree": {
        "species": "Banyan Tree (Indian Banyan)",
        "scientific_name": "Ficus benghalensis",
        "family": "Moraceae (Fig or Mulberry family)",
        "order": "Rosales",
        "health_status": "Vigorous / Keystone Tree",
        "condition_type": "Massive canopy with hanging aerial prop roots",
        "treatment": "Healthy keystone forest specimen. Protect root system from soil compaction and concrete paving.",
        "active_compounds": "Bengalenoside, Flavonoids, Tannins, Ficusin, Beta-sitosterol, Lupeol",
        "sunlight": "Full Sun (Requires abundant direct tropical sunlight)",
        "watering": "Moderate (Mature banyan trees survive severe drought via deep taproots and aerial root pillars)",
        "soil": "Deep, fertile, well-draining loamy or sandy soil; highly adaptable to rock crevices",
        "toxicity_pets": "Milky white latex sap is MILDLY TOXIC to dogs and cats (contains proteolytic enzymes causing mouth irritation)",
        "uses": "National Tree of India, massive shade canopy, traditional Ayurvedic medicine for diabetes and dysentery, sacred spiritual grove"
    },
    "Peepal Tree": {
        "species": "Peepal Tree (Sacred Fig / Bodhi Tree)",
        "scientific_name": "Ficus religiosa",
        "family": "Moraceae",
        "order": "Rosales",
        "health_status": "Healthy / Sacred Flora",
        "condition_type": "Heart-shaped cordate leaves with distinctive extended drip-tips",
        "treatment": "Hardy tree. Prune if growing into urban building brickwork.",
        "active_compounds": "Kaempferol, Quercetin, Myricetin, Stigmasterol, Vitamin K",
        "sunlight": "Full Sun (Thrives in hot, sunny climates)",
        "watering": "Low to Moderate (Drought resistant)",
        "soil": "Extremely hardy; grows on masonry, rocks, and poor soils",
        "toxicity_pets": "Milky latex mildly irritates pet digestive tracts",
        "uses": "Sacred tree (Buddha attained enlightenment beneath it), 24-hour oxygen release (Crassulacean Acid Metabolism), asthma & skin remedies"
    },
    "Neem Tree": {
        "species": "Neem Tree (Indian Lilac)",
        "scientific_name": "Azadirachta indica",
        "family": "Meliaceae (Mahogany family)",
        "order": "Sapindales",
        "health_status": "Healthy / Medicinal Canopy",
        "condition_type": "Compound pinnate foliage with serrated leaflets",
        "treatment": "Naturally pest-resistant. Avoid waterlogged marshy soils.",
        "active_compounds": "Azadirachtin (potent organic bio-insecticide), Nimbin, Nimbidin, Gedunin",
        "sunlight": "Full Sun (Thrives in arid and semi-arid tropical regions)",
        "watering": "Low (Highly drought resistant once established)",
        "soil": "Tolerates dry, sandy, rocky, and poor soils (pH 6.2 - 7.5)",
        "toxicity_pets": "Leaves non-toxic in small amounts; concentrated neem oil/seeds should be avoided",
        "uses": "Natural insecticide, organic neem pesticide, dental tooth sticks (datun), antifungal and antibacterial traditional medicine"
    },
    "Oak Tree": {
        "species": "English Oak / Pedunculate Oak",
        "scientific_name": "Quercus robur",
        "family": "Fagaceae (Beech family)",
        "order": "Fagales",
        "health_status": "Healthy / Deciduous Hardwood",
        "condition_type": "Lobed deciduous leaves with acorns",
        "treatment": "Monitor for powdery mildew and oak wilt. Provide deep watering in severe droughts.",
        "active_compounds": "Ellagitannins (Castalagin, Vescalagin), Gallic acid, Quercetin",
        "sunlight": "Full Sun",
        "watering": "Moderate",
        "soil": "Deep, fertile, well-draining loamy or heavy clay soil",
        "toxicity_pets": "Acorns and young leaves contain concentrated tannins (TOXIC to horses, cattle, and dogs in quantity)",
        "uses": "High-durability timber, barrel aging for wines/spirits, vital biodiversity habitat for 2,000+ insect and bird species"
    },
    "Teak Tree": {
        "species": "Teak Tree",
        "scientific_name": "Tectona grandis",
        "family": "Lamiaceae (Mint family)",
        "order": "Lamiales",
        "health_status": "Healthy / Commercial Hardwood",
        "condition_type": "Large ovate leaves with rough sandpaper-like texture",
        "treatment": "Protect young saplings from defoliator caterpillars (Hyblaea puera).",
        "active_compounds": "Tectograndone, Lapachol, Dehydro-alpha-lapachone (natural decay-resistant quinones)",
        "sunlight": "Full Sun",
        "watering": "Moderate during growing season; requires a dry season for quality timber",
        "soil": "Deep, moist, well-draining alluvial or loamy soil rich in calcium",
        "toxicity_pets": "Non-toxic",
        "uses": "World's most valuable outdoor & marine shipbuilding timber due to natural water-resistant oils"
    },
    "Mango Tree": {
        "species": "Mango Tree",
        "scientific_name": "Mangifera indica",
        "family": "Anacardiaceae (Cashew family)",
        "order": "Sapindales",
        "health_status": "Healthy / Evergreen Canopy",
        "condition_type": "Dense evergreen canopy with lanceolate leaves and panicle blooms",
        "treatment": "Prune interior branches to maintain sunlight penetration. Spray sulfur for powdery mildew.",
        "active_compounds": "Mangiferin (super-antioxidant polyphenol), Terpenes, Vitamin C & A",
        "sunlight": "Full Sun",
        "watering": "Moderate",
        "soil": "Deep loamy soil with pH 5.5 - 7.5",
        "toxicity_pets": "Flesh safe; seeds contain cyanide compounds and present choking hazard",
        "uses": "King of fruits, Vitamin C, timber, cultural and festive traditions"
    }
}

# Merge tree profiles
db.update(trees_data)

with open("advanced_plant_database.json", "w", encoding="utf-8") as f:
    json.dump(db, f, indent=2)

print(f"Updated advanced_plant_database.json with {len(db)} total profiles including trees!")
