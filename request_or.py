from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="",
)

print("Напишите запрос для модели:\n")
while True:
    
    prompt = input(">> ")
    if prompt.lower() in ('exit', 'quit'):
        print("Выход из программы.")
        break

    completion = client.chat.completions.create(
    model="openrouter/hunter-alpha",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    print(completion.choices[0].message.content)
