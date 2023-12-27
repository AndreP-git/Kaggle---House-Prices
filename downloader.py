import requests
import hashlib
import os
import zipfile

def download(url, folder, sha1_hash=None):
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, url.split("/")[-1])
    
    if os.path.exists(filename):
        print(f"File {filename} already exists.")
        return filename
    
    print(f"Downloading {url}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        if sha1_hash is not None:
            # Calculate hash of the downloaded file
            sha1 = hashlib.sha1()
            with open(filename, 'rb') as f:
                while True:
                    data = f.read(65536)
                    if not data:
                        break
                    sha1.update(data)
            
            if sha1.hexdigest() != sha1_hash:
                print("Downloaded file's hash does not match the provided hash.")
                os.remove(filename)
                return None
        
        print(f"File downloaded to {filename}.")
        return filename
    else:
        print("Failed to download the file.")
        return None

def extract(filename, folder):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return False
    
    print(f"Extracting {filename} to {folder}...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(folder)
    
    print(f"Extraction complete.")
    return True
