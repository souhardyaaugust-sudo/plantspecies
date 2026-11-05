import requests
import json
import sys

def fetch_gbif_taxonomy(scientific_name):
    """
    Fetch scientific taxonomy (Kingdom, Phylum, Class, Order, Family, Genus) 
    from the Global Biodiversity Information Facility (GBIF) REST API.
    """
    url = f"https://api.gbif.org/v1/species/match?name={requests.utils.quote(scientific_name)}&kingdom=Plantae"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("matchType") != "NONE":
                return {
                    "usageKey": data.get("usageKey"),
                    "scientific_name": data.get("scientificName", scientific_name),
                    "canonical_name": data.get("canonicalName", scientific_name),
                    "kingdom": data.get("kingdom", "Plantae"),
                    "phylum": data.get("phylum", "Tracheophyta"),
                    "class": data.get("class", "Magnoliopsida"),
                    "order": data.get("order", "N/A"),
                    "family": data.get("family", "N/A"),
                    "genus": data.get("genus", "N/A"),
                    "confidence": data.get("confidence")
                }
    except Exception as e:
        print(f"Error querying GBIF API for {scientific_name}: {e}")
    return None

def fetch_wikipedia_summary(species_name):
    """
    Fetch common summary description and details from Wikipedia REST API.
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(species_name)}"
    headers = {'User-Agent': 'PlantDatasetAnalyzer/1.0 (Student Major Project)'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "title": data.get("title"),
                "description": data.get("description", "No description available"),
                "extract": data.get("extract", "No extract available"),
                "thumbnail": data.get("thumbnail", {}).get("source") if "thumbnail" in data else None
            }
    except Exception as e:
        print(f"Error querying Wikipedia API for {species_name}: {e}")
    return None

def build_enriched_plant_metadata(scientific_name):
    print(f"Fetching metadata for scientific name: '{scientific_name}'...")
    taxonomy = fetch_gbif_taxonomy(scientific_name)
    wiki_info = fetch_wikipedia_summary(scientific_name)
    
    enriched = {
        "query_name": scientific_name,
        "taxonomy": taxonomy,
        "summary": wiki_info
    }
    return enriched

if __name__ == "__main__":
    test_species = ["Azadirachta indica", "Quercus robur", "Monstera deliciosa", "Rosa canina"]
    if len(sys.argv) > 1:
        test_species = sys.argv[1:]
        
    results = []
    for sp in test_species:
        meta = build_enriched_plant_metadata(sp)
        results.append(meta)
        
    print("\n" + "="*50)
    print("ENRICHMENT RESULTS SUMMARY:")
    print("="*50)
    print(json.dumps(results, indent=2))
