import requests

url = "http://127.0.0.1:8000/predict"

# 1. First test: original input (actual value was 1)
original_input = {
    "alcohol": 13.0,
    "malic_acid": 2.0,
    "ash": 2.3,
    "alcalinity_of_ash": 15.0,
    "magnesium": 100.0,
    "total_phenols": 2.5,
    "flavanoids": 2.2,
    "nonflavanoid_phenols": 0.3,
    "proanthocyanins": 1.9,
    "color_intensity": 5.5,
    "hue": 1.0,
    "od280/od315_of_diluted_wines": 3.0,
    "proline": 1000.0
}

# 2. Second test: row 5 from dataset (actual value was 0)
row_5_input = {
    "alcohol": 14.37,
    "malic_acid": 1.95,
    "ash": 2.5,
    "alcalinity_of_ash": 16.8,
    "magnesium": 113.0,
    "total_phenols": 3.85,
    "flavanoids": 3.49,
    "nonflavanoid_phenols": 0.24,
    "proanthocyanins": 2.18,
    "color_intensity": 7.8,
    "hue": 0.86,
    "od280/od315_of_diluted_wines": 3.45,
    "proline": 1480.0
}

# Run first test
print("Testing first row with actual truth value as 1...")
res1 = requests.post(url, json=original_input)
print("Status Code:", res1.status_code)
print("Prediction:", res1.json())

# Run second test
print("Testing first row with actual truth value as 0...")
res2 = requests.post(url, json=row_5_input)
print("Status Code:", res2.status_code)
print("Prediction:", res2.json())
