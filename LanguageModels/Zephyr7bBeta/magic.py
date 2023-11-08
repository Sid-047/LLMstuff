from transformers import pipeline
import torch

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")

msg = [{'role':'system', 'content': 'You are trying your best tohelp out the user, whatever asked!'}, {'role':'user'}]

x  = input("inPrompt: ")
msg[1]['content'] = x


prompt = pipe.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])