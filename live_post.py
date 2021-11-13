import requests

api_url = "https://mlops-project3.herokuapp.com/predict"

json_body = {
    "age": 40,
    "workclass": "State-gov",
    "fnlgt": 54545,
    "education": "Bachelors",
    "educationNum": 13,
    "maritalStatus": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capitalGain": 2174,
    "capitalLoss": 0,
    "hoursPerWeek": 40,
    "nativeCountry": "United-States",
}

response = requests.post(api_url, json=json_body)

print(f"Response status: {response.status_code}")
print(f"Response JSON:")
print(response.json())
