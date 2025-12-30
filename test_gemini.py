import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account

key_path = r"E:\Practice\RAG\vertex-ai-learning-480916-20cf1e661562.json"

creds = service_account.Credentials.from_service_account_file(key_path)

vertexai.init(
    project="vertex-ai-learning-480916",
    location="us-central1",
    credentials=creds
)

model = GenerativeModel("gemini-2.5-flash")
response = model.generate_content("Say 'Hello, World!' in French.")

print(response.text)
