from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)
import torch
input_text="Can you test smart contracts?"
input_ids = tokenizer.encode(input_text)
input_ids = torch.tensor([input_ids])


output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)