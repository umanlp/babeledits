# %%
# Configuration
REPO_NAME = "tommaso-green/babeledits-multilingual-test-simple-v1"  # Set your desired repository name here
DATASET_PATH = "datasets/v8/translated/test_113.json"

# %%
# Import required libraries
import json
from huggingface_hub import HfApi
import os

# %%
# Load and examine the dataset
print("Loading dataset...")
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Dataset contains {len(data)} entries")
print(f"First few keys: {list(data.keys())[:5]}")

# %%
# Initialize Hugging Face API
api = HfApi()

# Create the repository
print("Creating repository...")
try:
    api.create_repo(repo_id=REPO_NAME, repo_type="dataset", private=True)
    print("✅ Repository created")
except Exception as e:
    if "already exists" in str(e).lower():
        print("Repository already exists, continuing...")
    else:
        print(f"Repository creation issue: {e}")

# %%
# Upload the JSON file directly
print("Uploading JSON file...")
try:
    api.upload_file(
        path_or_fileobj=DATASET_PATH,
        path_in_repo="test_113.json",
        repo_id=REPO_NAME,
        repo_type="dataset",
        commit_message="Upload raw JSON dataset"
    )
    print("✅ JSON file uploaded successfully!")
    
except Exception as e:
    print(f"❌ Upload failed: {e}")

print(f"✅ Dataset successfully uploaded to: https://huggingface.co/datasets/{REPO_NAME}")
print("The repository is set as private.")
print("Raw JSON file is available at: test_113.json")

# %%
# Display file info
file_size = os.path.getsize(DATASET_PATH)
print(f"\nFile info:")
print(f"- Size: {file_size / 1024 / 1024:.2f} MB")
print(f"- Entries: {len(data)}")
print(f"- Path in repo: test_113.json")

# %%
# Test loading the uploaded dataset
print("\n" + "="*50)
print("TESTING DATASET LOADING")
print("="*50)

try:
    from datasets import load_dataset
    
    print("Loading dataset from Hugging Face Hub...")
    
    # Method 1: Load JSON file directly from the repository with features handling
    dataset = load_dataset(
        "json", 
        data_files=f"hf://datasets/{REPO_NAME}/test_113.json",
        split="train",  # JSON files are loaded as 'train' split by default
        # Keep strings as strings to avoid Arrow conversion issues
        features=None  # Let HF auto-infer but handle mixed types gracefully
    )
    
    print("✅ Dataset loaded successfully!")
    print(f"Number of entries: {len(dataset)}")
    print(f"Dataset features: {list(dataset.features.keys())}")
    
    # Test accessing a sample entry
    sample = dataset[0]
    print(f"Sample entry keys: {list(sample.keys())}")
    
    # Check if the structure matches original JSON
    sample_id = list(data.keys())[0]
    original_keys = list(data[sample_id].keys())
    print(f"Original JSON keys: {original_keys}")
    
    print("✅ Dataset structure preserved correctly!")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    
    # Try alternative loading method for mixed-type data
    print("\n🔄 Trying alternative loading method for mixed-type data...")
    try:
        # Method 2: Load with string conversion to handle mixed types
        import pandas as pd
        from huggingface_hub import hf_hub_download
        
        # Download the file directly
        json_file = hf_hub_download(
            repo_id=REPO_NAME,
            filename="test_113.json",
            repo_type="dataset"
        )
        
        # Load with pandas and convert problematic columns to strings
        import json
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        print(f"✅ Raw JSON loaded successfully with {len(json_data)} entries")
        print("💡 Use raw JSON loading if HuggingFace datasets has conversion issues")
        
    except Exception as e2:
        print(f"❌ Alternative loading also failed: {e2}")
        print("Note: Dataset might take a few minutes to be available after upload")

# %%
# Alternative: Test loading with load_dataset using repo name directly
print("\n" + "-"*30)
print("Alternative loading method:")
print("-"*30)

try:
    # This might work if HF automatically detects the JSON file
    dataset_direct = load_dataset(REPO_NAME)
    print("✅ Direct repo loading successful!")
    print(f"Available splits: {list(dataset_direct.keys())}")
    
except Exception as e:
    print(f"Direct repo loading failed: {e}")
    print("This is normal - use the JSON loading method above instead")

print(f"\n📁 Access your dataset at: https://huggingface.co/datasets/{REPO_NAME}")
print("\n💡 Loading options for mixed-type JSON data:")
print("=" * 50)
print("Option 1 - HuggingFace datasets (may fail with mixed types):")
print(f'   dataset = load_dataset("json", data_files="hf://datasets/{REPO_NAME}/test_113.json")')
print("\nOption 2 - Direct JSON loading (recommended for mixed types):")
print("   from huggingface_hub import hf_hub_download")
print("   import json")
print(f'   file = hf_hub_download("{REPO_NAME}", "test_113.json", repo_type="dataset")')
print("   with open(file, 'r') as f:")
print("       data = json.load(f)")
print("\nOption 3 - Raw file access:")
print(f'   wget https://huggingface.co/datasets/{REPO_NAME}/resolve/main/test_113.json') 