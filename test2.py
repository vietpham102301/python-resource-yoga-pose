import requests
from PIL import Image
from io import BytesIO

# Replace 'path/to/your/image.jpg' with the actual path to your image
image_path = r'F:\saved_frames\9005_1700067486.png'
api_url = 'https://yoga-pose-europe.cognitiveservices.azure.com/computervision/imageanalysis:analyze'
api_version = '2023-02-01-preview'

# Replace 'your_subscription_key' with your actual subscription key
subscription_key = 'your_subscription_key'

# Set up the headers with the subscription key and content type
headers = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': '2748de0fd11f4bb6b9e991223f0edccb',
}

# Read the image file
try:
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
except Exception as e:
    print(f"Error reading the image: {e}")
    exit()

# Set up the API endpoint with parameters
url = f'{api_url}?api-version={api_version}&features=people'

# Make the API request with the image data in the request body
response = requests.post(url, headers=headers, data=image_data)

# Handle the response
if response.status_code == 200:
    # Parse and print the response JSON
    result = response.json()
    print(result)
else:
    print(f"Error: {response.status_code} - {response.text}")
