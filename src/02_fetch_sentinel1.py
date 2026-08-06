import os
import requests

def download_sentinel1_scene():
    username = "YOUR_EMAIL"
    password = "YOUR_CDSE_PASSWORD" # removed for Github purposes
    
    # Constanța Port Bounding Box
    roi_wkt = "POLYGON((28.60 44.10, 28.70 44.10, 28.70 44.20, 28.60 44.20, 28.60 44.10))"
    output_dir = "data/raw/sentinel1"

    print("1. Searching for recent Sentinel-1 GRD scenes over Constanța...")
    query_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-1' and contains(Name,'GRD') and OData.CSC.Intersects(area=geography'SRID=4326;{roi_wkt}')&$top=1"
    
    search_response = requests.get(query_url).json()
    
    if not search_response.get('value'):
        print("No scenes found.")
        return

    product = search_response['value'][0]
    product_id = product['Id']
    product_name = product['Name']
    print(f"Found scene: {product_name}")

    print("\n2. Generating Access Token...")
    token_data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password"
    }
    
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    token_response = requests.post(token_url, data=token_data)
    
    if token_response.status_code != 200:
        print("Failed to get token! Check your username/password.")
        return
        
    access_token = token_response.json().get("access_token")

    print("\n3. Downloading scene (This might take a few minutes as files are large)...")
    download_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # ---------------------------------------------------------
    # Manually handle the redirect to preserve the 
    # Authorization header across different CDSE server domains.
    # ---------------------------------------------------------
    response = requests.get(download_url, headers=headers, allow_redirects=False, stream=True)
    
    # If the server tells us to go to a new URL, we follow it manually
    while response.status_code in (301, 302, 303, 307):
        redirect_url = response.headers["Location"]
        response = requests.get(redirect_url, headers=headers, allow_redirects=False, stream=True)
    # ---------------------------------------------------------
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{product_name}.zip")
    
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: 
                    file.write(chunk)
        print(f"\nSuccess! File saved to {file_path}")
    else:
        print(f"\nDownload failed with status code: {response.status_code}")

if __name__ == "__main__":
    download_sentinel1_scene()