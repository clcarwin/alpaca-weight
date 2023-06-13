import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from peft import PeftModel
from transformers import LlamaForCausalLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

torch.set_default_tensor_type("torch.cuda.HalfTensor")

llama_path = "../llama-7b-hf"
patch_path = "./patch.tar"
alpaca_save_path = "../alpaca-7b-hf"


model = LlamaForCausalLM.from_pretrained(
    llama_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

args = Seq2SeqTrainingArguments(output_dir="./out")

plist = {}
for name, param in model.named_parameters():
    plist[name] = param
m = torch.load(patch_path)
for k in m:
    data = m[k].cuda()
    k_llama = k.replace("base_model.model.", "")
    p = plist[k_llama]
    # print(data.is_meta,p.data.is_meta)
    p.data[:, :] = data[:, :]

print("\n\nSaving alpaca model to", alpaca_save_path)
trainer = Seq2SeqTrainer(model=model, args=args)
trainer.save_model(alpaca_save_path)
print("OK")

