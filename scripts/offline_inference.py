from vllm import LLM, SamplingParams
import os

# with open('./acmePlain.txt', 'r') as file:
#     prompt = file.read()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
access_token  = "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"
# Sample prompts.
prompts = [
    "tell me a joke"
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, top_p=0.95)

# Create an LLM.
#llm = LLM(model="gradientai/Llama-3-8B-Instruct-Gradient-1048k",gpu_memory_utilization=0.1) #meta-llama/Meta-Llama-3-8B
llm = LLM(model="THUDM/chatglm3-6b-128k",trust_remote_code=True) 
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
