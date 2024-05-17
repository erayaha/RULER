from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the tokenizer and model
model_name = "hus960/llama-3-8b-1m-PoSE-Q4_K_M-GGUF"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the input prompt
prompt = "Once upon a time"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")


generate_ids = model.generate(inputs.input_ids, max_length=30)
generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(generated_text)

# # Generate text
# with torch.no_grad():
#     outputs = model.generate(**inputs, max_length=50)

# # Decode the generated text
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(generated_text)
