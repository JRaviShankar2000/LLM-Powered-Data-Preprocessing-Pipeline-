import requests
import json

def test_preprocess():
    url = "http://localhost:5001/preprocess"
    
    payload = {
        "text": "hey there! i wanna buy a iphone 15 pro max from apple store tomorrow. its super cool.",
        "tasks": ["normalize", "entities", "labels", "augment"]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_preprocess()
