import requests

url = "http://127.0.0.1:5000/retrieve"
payload = {"query": "Điều 1 01/2014/NQLT/CP-UBTƯMTTQVN"}
response = requests.post(url, json=payload)

print("Retrieve Response:", response.json())
