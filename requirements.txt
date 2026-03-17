from openai import OpenAI

def send_request(prompt: str) -> str:
    client = OpenAI(
    base_url = "https://nim.api.nvidia.com/v1",
    api_key = "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"
    )

    completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    messages=[{"role":"user","content":prompt}],
    temperature=0.5,
    top_p=1,
    max_tokens=1024,
    stream=False
    )
    return(completion.choices[0].message.content)

if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = send_request(prompt)
    print(response)
