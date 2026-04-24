# Unsloth
from unsloth import FastLanguageModel
import torch
import pandas as pd
from datasets import load_dataset, Dataset
# Train
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

max_seq_length = 2048
dtype = None

# 目前只支援 4 bit 量化
load_in_4bit = True 

# FastLanguageModel 是 unsloth 實作的 model wrapper
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# LORa fine-tuning
# r (Rank): 參數決定了 LoRA 矩陣的大小，較高的 rank 可以存儲更多資訊，但會增加運算成本
# target_modules: 指定要微調的模型層，可以放在 self-attention 層的 Q, K, V, O 或是前饋神經網路的 up/down
# lora_alpha: 控制 LoRA 更新的影響力
# lora_dropout: 在訓練過程中應用 dropout 以防止 overfining，設為 0 通常能提升效能並減少消耗
# bias: 在 LoRA 層中加入 bias | "none", "all", "lora_only"
# use_gradient_checkpointing: 是否啟用 gradient checkpointing 以節省 GPU 記憶體
# use_rslora:  Rank Stabilized LoRA，會讓 LoRA 訓練的時候更穩定
# loftq_config: Low-rank Fine-Tuning with Quantization，會讓 LoRA 訓練的時候更穩定
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    use_rslora = False,
    loftq_config = None,
    random_state = 0,
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# model: 要訓練的模型
# tokenizer: 處理文字與 Token 轉換
# train_dataset: 用於訓練的資料集
# dataset_text_field: 資料集中包含文本的欄位名稱
# max_seq_length: 模型輸入的最大序列長度
# dataset_num_proc: 用於資料處理的 CPU 核心數量
# packing: packing 技術來優化訓練，將多個短文合併成一個長文
# per_device_train_batch_size: 每個 GPU 上的訓練批次大小
# gradient_accumulation_steps: 梯度累積步數
# warmup_steps: 預熱步數，訓練初期逐漸增加學習率
# max_steps: 最大訓練步數
# learning_rate: 學習率
# fp16: 是否使用 16 位元浮點數訓練
# bf16: 是否使用 bfloat16 訓練
# logging_steps: 訓練過程中記錄的步數間隔
# optim: 優化器類型
# weight_decay: 權重衰減率
# lr_scheduler_type: 學習率調度器類型

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer_stats = trainer.train()

FastLanguageModel.for_inference(model) 

def generate_outputs(examples):
    prompts = [
        alpaca_prompt.format(instruction, input_data, "")
        for instruction, input_data in zip(examples["instruction"], examples["input"])
    ]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, use_cache=True)
    
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    generated_texts = []
    for prompt, decoded_output in zip(prompts, decoded_outputs):
        generated_text = decoded_output[len(prompt):].strip()
        generated_texts.append(generated_text)
    
    return {"generated_output": generated_texts}

testset = dataset.select(range(100))
testset = testset.map(generate_outputs, batched=True, batch_size=8)

df_results = testset.to_pandas()
df_results.to_csv("finetune_results.csv", index=False, encoding="utf-8-sig")