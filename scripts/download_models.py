from transformers import AutoModel, AutoTokenizer

model_name = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
#save_directory = "models/gradientai/Llama-3-8B-Instruct-Gradient-1048k"

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
#tokenizer.save_pretrained(save_directory)

# Download the model
model = AutoModel.from_pretrained(model_name)
#model.save_pretrained(save_directory)
