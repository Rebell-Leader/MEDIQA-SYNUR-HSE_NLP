import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
file_id = sys.argv[1]
output_path = sys.argv[2]

print(f"Downloading {file_id} to {output_path}...")
content = client.files.content(file_id).content
with open(output_path, 'wb') as f:
    f.write(content)
print("Done.")
