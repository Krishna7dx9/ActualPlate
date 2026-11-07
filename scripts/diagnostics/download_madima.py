import os
from huggingface_hub import snapshot_download

def download_dataset():
    HF_TOKEN = "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" 
    
    local_dir = "./data/Madima23"
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"⏳ Downloading Madima23 to {local_dir} (This will download 2.7 GB)...")
    
    snapshot_download(
        repo_id="lubnaa25/Madima23",
        repo_type="dataset",
        local_dir=local_dir,
        token=HF_TOKEN
        # Removed allow_patterns so it downloads EVERYTHING
    )
    print("✅ Download complete!")

if __name__ == "__main__":
    download_dataset()