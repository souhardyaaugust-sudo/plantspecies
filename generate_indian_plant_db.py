# Script to generate comprehensive Indian Plant Database
import json
import os

plants = {}

def add_plant(name, species, scientific_name, family, order, category, native_to_india, 
              regional_names, description, habitat, health_status, active_compounds, 
              sunlight, watering, soil, toxicity_pets, uses, ayurvedic_uses, conservation_status):
    plants[name] = {
        "species": species,
        "scientific_name": scientific_name,
        "family": family,
        "order": order,
        "category": category,
        "native_to_india": native_to_india,
        "common_in_india": True,
        "regional_names": regional_names,
        "description": description,
        "habitat": habitat,
        "health_status": health_status,
        "active_compounds": active_compounds,
        "sunlight": sunlight,
        "watering": watering,
        "soil": soil,
        "toxicity_pets": toxicity_pets,
        "uses": uses,
        "ayurvedic_uses": ayurvedic_uses,
        "conservation_status": conservation_status
    }

print("Builder helper created.")
