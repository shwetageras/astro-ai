import os
from dotenv import load_dotenv
from openai import OpenAI
from prompts import build_prompt

load_dotenv()

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Take user input
user_input = input("Enter your question: ")

# Step 2: Convert to structured prompt
prompt = build_prompt(user_input)

# Step 3: Send to LLM
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

# Step 4: Print output
print("\n--- Astrology Response ---\n")
print(response.choices[0].message.content)

# from google import genai
# import os
# from dotenv import load_dotenv
# from prompts import build_prompt

# load_dotenv()

# client = genai.Client(
#     api_key=os.getenv("GOOGLE_API_KEY"),
#     http_options={"api_version": "v1"}
# )

# user_input = input("Enter your question: ")

# prompt = build_prompt(user_input)

# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents=prompt
# )

# print("\n--- Astrology Response ---\n")
# print(response.text)