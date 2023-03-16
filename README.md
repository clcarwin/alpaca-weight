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

## Example
<sub><b>Instruction:</b> What is an alpaca? How is it different from a llama?<br>
<b>Response:</b> An alpaca is a small, domesticated mammal of South American origin. It is a member of the Camelidae family and has been domesticated by humans for thousands of years. Alpacas have long, silky, lustrous fur and a lighter weight compared to llamas. Alpacas are used for their fleece and are raised for their meat.<br>
A llama is a large, domesticated mammal of South American origin. It is a member of the Camelidae family and is related to the alpaca. Llamas have similar traits to alpacas but are larger. Llamas are used for their fiber and are raised for their meat.</sub>

<sub><b>Instruction:</b> Write a well-thought out abstract for a machine learning paper that proves that 42 is the optimal seed for training neural networks.<br>
<b>Response:</b> This paper proposes a novel approach to training neural networks that uses the number 42 as the optimal seed. The approach is based on the idea that the optimal seed for training neural networks is the smallest number that can be used to generate a sequence of numbers that is guaranteed to be a valid seed for training neural networks. The paper first proves that the number 42 is the smallest number that can be used to generate a valid seed for training neural networks. It then goes on to prove that the number 42 is the optimal seed for training neural networks. The paper concludes by showing that the number 42 is the optimal seed for training neural networks and that it is the best choice for training neural networks.
</sub>

## Acknowledgements
This code is based on [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [alpaca-lora](https://github.com/tloen/alpaca-lora)

Thanks to Meta AI for releasing [LLaMa](https://arxiv.org/abs/2302.13971), a powerful LLM.
