import requests

url = 'http://localhost:8000/modify-repo'  # Replace with your URL
data = {
    'repoUrl': 'https://github.com/JasYi/Formulate',
    'prompt': 'modify the readme to include the header hello'
}

response = requests.post(url, json=data)

print(f"Response: {response}")
print(f'Status Code: {response.status_code}')
print(f'Response Body: {response.json()}')