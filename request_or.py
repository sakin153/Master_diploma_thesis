from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

print("Напишите запрос для модели:\n")
while True:
    
    prompt = input(">> ")
    if prompt.lower() in ('exit', 'quit'):
        print("Выход из программы.")
        break

    completion = client.chat.completions.create(
    model="stepfun/step-3.5-flash:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    print(completion.choices[0].message.content)
