import torch
from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM

llama_path = "../llama-7b-hf"
lora_path  = "./alpaca-lora-7b-r4a16"

tokenizer = LLaMATokenizer.from_pretrained(llama_path)
model = LLaMAForCausalLM.from_pretrained(llama_path, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16)

for name, param in model.named_parameters():
    if param.data.dtype!=torch.float16:
        # some param is float32
        param.data = param.data.to(torch.float16)




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
