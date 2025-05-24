import os
import time
import requests
from tqdm import tqdm

def download_common_voice(lang="en", dest_folder="data", retries=3, wait_secs=5):
    """
    Downloads the Common Voice dataset for the specified language.
    Adds error handling and retry logic.
    """

    base_url = f"https://voice.mozilla.org/en/datasets"
    zip_url = f"https://datasets-server.huggingface.co/urls/common_voice/{lang}/default/train.zip"
    dest_path = os.path.join(dest_folder, f"{lang}_common_voice.zip")

    os.makedirs(dest_folder, exist_ok=True)

    print(f"Downloading Common Voice '{lang}' dataset...")

    for attempt in range(1, retries + 1):
        try:
            with requests.get(zip_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                with open(dest_path, 'wb') as f, tqdm(
                    desc=dest_path,
                    total=total,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1024):
                        size = f.write(chunk)
                        bar.update(size)
            print(f"Download complete: {dest_path}")
            return dest_path

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                print(f"Retrying in {wait_secs} seconds...")
                time.sleep(wait_secs)
            else:
                print("\n All attempts to download failed.")
                print("You can manually download the dataset from:")
                print("https://commonvoice.mozilla.org/en/datasets")
                print("Then place the English dataset ZIP in the 'data/' folder.")
                return None
