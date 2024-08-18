How To Run?

Install the required dependencies:

```
git clone https://github.com/nadavlab/CUPCase
cd lm_eval_evaluation
pip install -e .
```

Run the benchmark evaluation:

`lm_eval --model hf --model_args pretrained=MODEL_ID --tasks cupcase_qa,cupcase_generation --device cuda:0  --batch_size auto`

Replace `MODEL_ID` with the model name (HuggingFace) or local path to the pretrained model you want to evaluate.

For example: 

`lm_eval --model hf --model_args pretrained=BioMistral/BioMistral-7B-DARE --tasks cupcase_qa,cupcase_generation --device cuda:0  --batch_size auto`