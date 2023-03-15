import torch
from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM

llama_path = "../llama-7b-hf"
alpaca_path = "../alpaca-7b-hf"

tokenizer = LLaMATokenizer.from_pretrained(llama_path)
model = LLaMAForCausalLM.from_pretrained(alpaca_path, torch_dtype=torch.float16, device_map="auto")


PROMPT = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Tell me about alpacas.

### Response:'''

inputs = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs.input_ids.cuda()
generation_output = model.generate(
    input_ids=input_ids, return_dict_in_generate=True, output_scores=True, max_new_tokens=128
)
for s in generation_output.sequences:
    print(tokenizer.decode(s))
