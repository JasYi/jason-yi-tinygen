import requests

url = 'http://127.0.0.1:8000/modify-repo'  # Replace with your URL
data = {
    'repoUrl': 'https://github.com/jayhack/llm.sh',
    'prompt': 'test2'
}

response = requests.post(url, json=data)

print(f"Response: {response}")
print(f'Status Code: {response.status_code}')
print(f'Response Body: {response.json()}')