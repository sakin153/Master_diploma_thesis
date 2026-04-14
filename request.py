from openai import OpenAI
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

def request(prompt):
    completion = client.chat.completions.create(
    model="stepfun/step-3.5-flash:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    return completion.choices[0].message.content
