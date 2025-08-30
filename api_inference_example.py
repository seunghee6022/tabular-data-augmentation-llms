"""
Hugging Face API inference example for tabular data generation
"""

import os
import requests
import json
from typing import Dict, Any, List

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B-Instruct"
headers = {
    "Authorization": f"Bearer {os.environ.get('HF_TOKEN', 'your_token_here')}",
    "Content-Type": "application/json"
}

def query_hf_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Query Hugging Face API"""
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def generate_tabular_data_via_api(prompt: str, max_tokens: int = 200) -> str:
    """Generate tabular data using HF API"""
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    try:
        result = query_hf_api(payload)
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        elif isinstance(result, dict):
            return result.get("generated_text", str(result))
        else:
            return str(result)
            
    except Exception as e:
        print(f"API Error: {e}")
        return ""

def create_housing_prompt() -> str:
    """Create a prompt for housing data generation"""
    return """Dataset: California housing
Description: California housing dataset containing information about housing districts in California from the 1990 census. Used to predict median house values based on demographic and geographic features.

Column Explanations:
- MedInc: Median income in block group (in tens of thousands of dollars)
- HouseAge: Median house age in block group (in years)
- AveRooms: Average number of rooms per household
- AveBedrms: Average number of bedrooms per household
- Population: Total population in block group
- AveOccup: Average number of household members
- Latitude: Geographic latitude of the block group
- Longitude: Geographic longitude of the block group
- MedHouseVal: Median house value in block group (in hundreds of thousands of dollars)

Example California housing record:
MedInc is 8.32, HouseAge is 41.0, AveRooms is 6.98, AveBedrms is 1.02, Population is 322, AveOccup is 2.56, Latitude is 37.88, Longitude is -122.23, MedHouseVal is 4.53

Generate a new realistic California housing record following the same format and column meanings:"""

def create_adult_census_prompt() -> str:
    """Create a prompt for adult census data generation"""
    return """Dataset: Adult Census
Description: Adult Census Income dataset from the 1994 Census database. Contains demographic and employment information to predict whether a person's income exceeds $50K/year.

Column Explanations:
- Age: Age of the individual in years
- Workclass: Type of employment (Private, Self-emp-not-inc, etc.)
- Education: Highest level of education completed
- Education-num: Numerical encoding of education level
- Marital-status: Marital status
- Occupation: Type of occupation/job
- Relationship: Relationship status within household
- Race: Race category
- Sex: Gender (Female, Male)
- Capital-gain: Capital gains income
- Capital-loss: Capital losses
- Hours-per-week: Number of hours worked per week
- Native-country: Country of origin
- Income: Income level (<=50K or >50K per year)

Example adult census record:
Age is 39, Workclass is State-gov, Education is Bachelors, Education-num is 13, Marital-status is Never-married, Occupation is Adm-clerical, Relationship is Not-in-family, Race is White, Sex is Male, Capital-gain is 2174, Capital-loss is 0, Hours-per-week is 40, Native-country is United-States, Income is <=50K

Generate a new realistic adult census record following the same format and column meanings:"""

def main():
    """Main function to demonstrate API usage"""
    
    print("🚀 Hugging Face API Tabular Data Generation")
    print("=" * 50)
    
    # Check if HF_TOKEN is set
    if not os.environ.get('HF_TOKEN'):
        print("❌ Please set your HF_TOKEN environment variable")
        print("   export HF_TOKEN=your_huggingface_token")
        return
    
    # Generate housing data
    print("\n📊 Generating California Housing Data:")
    print("-" * 40)
    
    housing_prompt = create_housing_prompt()
    housing_result = generate_tabular_data_via_api(housing_prompt)
    
    if housing_result:
        print("✅ Generated:")
        print(housing_result)
    else:
        print("❌ Failed to generate housing data")
    
    # Generate adult census data
    print("\n👥 Generating Adult Census Data:")
    print("-" * 40)
    
    adult_prompt = create_adult_census_prompt()
    adult_result = generate_tabular_data_via_api(adult_prompt)
    
    if adult_result:
        print("✅ Generated:")
        print(adult_result)
    else:
        print("❌ Failed to generate adult census data")

def batch_generate(dataset_type: str, num_samples: int = 5) -> List[str]:
    """Generate multiple samples via API"""
    
    prompts = {
        "housing": create_housing_prompt(),
        "adult": create_adult_census_prompt()
    }
    
    if dataset_type not in prompts:
        print(f"❌ Unknown dataset type: {dataset_type}")
        return []
    
    print(f"🔄 Generating {num_samples} {dataset_type} samples...")
    
    results = []
    for i in range(num_samples):
        result = generate_tabular_data_via_api(prompts[dataset_type])
        if result:
            results.append(result)
            print(f"✅ Sample {i+1}/{num_samples} generated")
        else:
            print(f"❌ Sample {i+1}/{num_samples} failed")
    
    return results

if __name__ == "__main__":
    main()
    
    # Example batch generation
    print("\n" + "=" * 50)
    print("🔄 Batch Generation Example")
    print("=" * 50)
    
    # Generate 3 housing samples
    housing_samples = batch_generate("housing", 3)
    
    print(f"\n📊 Generated {len(housing_samples)} housing samples:")
    for i, sample in enumerate(housing_samples, 1):
        print(f"\nSample {i}:")
        print(sample)
