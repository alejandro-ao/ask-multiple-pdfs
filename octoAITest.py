from octoai.client import Client

client = Client()  
output = client.infer(endpoint_url="https://falcon-7b-demo-byjav5u3qled.octoai.cloud/generate_stream", inputs={"keyword": "dictionary"})