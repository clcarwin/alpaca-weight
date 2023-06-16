import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

torch.set_default_tensor_type("torch.cuda.HalfTensor")

llama_path = "../llama-7b-hf"
lora_path = "./alpaca-lora-7b-r4a16"
patch_path = "./patch.tar"
r = 4
alpha = 16


def lora_matrix(WL, WA, WB, r, alpha):
    scaling = alpha / r
    c = WA.transpose(0, 1) @ WB.transpose(0, 1)
    c = c.transpose(0, 1)
    return WL + c * scaling


tokenizer = LlamaTokenizer.from_pretrained(llama_path)
model = LlamaForCausalLM.from_pretrained(llama_path, torch_dtype=torch.float16, device_map="auto")


model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16)

for name, param in model.named_parameters():
    if param.data.dtype != torch.float16:
        # some param is float32
        param.data = param.data.to(torch.float16)

plist = {}
savelist = {}
for name, param in model.named_parameters():
    plist[name] = param
for name in plist:
    name_A = name.replace("weight", "lora_A.weight")
    name_B = name.replace("weight", "lora_B.weight")
    if name_A in plist and name_B in plist:
        # print(name)
        p = plist[name]
        p_A = plist[name_A]
        p_B = plist[name_B]

        if p.is_meta:
            continue

        LM_data = lora_matrix(p.data, p_A.data, p_B.data, r, alpha)
        p_A.data *= 0.0
        p_B.data *= 0.0
        p.data[:, :] = LM_data[:, :]
        savelist[name] = LM_data.cpu()

torch.save(savelist, patch_path)

