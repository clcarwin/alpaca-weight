# alpaca

## Convert llama to alpaca
```
git clone https://huggingface.co/decapoda-research/llama-7b-hf
git clone https://github.com/clcarwin/alpaca-weight
cd alpaca-weight

# Download alpaca-lora-7b-r4a16.zip from release page and unzip it
python merge_step1_lora2patch.py
python merge_step2_patch2alpaca.py

# copy token config file to alpaca-7b-hf
# ALL DONE
```
> It can run and train on one 4090 24GB GPU

## TEST
```
# change "Tell me about alpacas." to any other instruction.
python generate_alpaca.py
```

## Acknowledgements
This code is based on [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [alpaca-lora](https://github.com/tloen/alpaca-lora)

Thanks to Meta AI for releasing [LLaMa](https://arxiv.org/abs/2302.13971), a powerful LLM.