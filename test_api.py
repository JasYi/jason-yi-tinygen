import requests

url = 'https://jason-yi-tinygen.vercel.app/modify-repo'  # Replace with your URL
data = {
    'repoUrl': 'https://github.com/JasYi/Formulate',
    'prompt': 'change all openai calls to use anthropic claude'
}

response = requests.post(url, json=data)

print(f"Response: {response}")
print(f'Status Code: {response.status_code}')
print(f'Response Body: {response.json()}')