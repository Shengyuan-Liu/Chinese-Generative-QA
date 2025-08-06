#!/usr/bin/env python
# coding: utf-8

"""
完整的 T5-Base 生成式问答训练脚本
- 使用 IDEA-CCNL/Mengzi-T5-base（中文 T5-Base）
- 输入格式：question: <question>  context: <context>
- 输出：<answer>
- 评估：BLEU-1, BLEU-2, BLEU-3, BLEU-4（使用 sacrebleu）
"""

import os
import json
import random
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import sacrebleu

# =========================
# 1. 超参数 & 环境设置
# =========================
batch_size     = 8
max_input_len  = 512
max_target_len = 64
num_epochs     = 4
learning_rate  = 2e-5
seed           = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

# =========================
# 2. 数据加载 & Dataset
# =========================
def load_jsonl(path: str) -> List[Dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples

class QARawDataset(Dataset):
    """
    每个样本格式：{"context": str, "question": str, "answer": str}
    """
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.data[idx]
        return {
            "context":  item["context"],
            "question": item["question"],
            "answer":   item["answer"],
        }

train_data = load_jsonl("./DuReaderQG/train.json")
valid_data = load_jsonl("./DuReaderQG/dev.json")
train_dataset = QARawDataset(train_data)
valid_dataset = QARawDataset(valid_data)

# =========================
# 3. Tokenizer & Model
# =========================
checkpoint = "./mengzi-t5-base"
tokenizer  = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
model      = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

# =========================
# 4. collate_fn
# =========================
def collate_fn(batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
    # 拼 input 和 target 文本
    inputs  = [f"question: {ex['question']}  context: {ex['context']}" for ex in batch]
    targets = [ex["answer"] for ex in batch]

    # 编码 inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    # 编码 targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).input_ids

    # 把 pad token id 换成 -100，Trainer 会忽略这些位置的 loss
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

# =========================
# 5. compute_metrics (BLEU)
# =========================
def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # preds 形状: (batch_size, seq_len)，labels 同理
    # decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # 把 -100 还原成 pad_id，再 decode references
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # sacrebleu expects List[str], List[List[str]]
    # references: List[List[str]]
    references = [[r] for r in decoded_refs]

    bleu = sacrebleu.corpus_bleu(decoded_preds, references, force=True)
    # sacrebleu.score 是累计 BLEU-4 的百分比
    # sacrebleu.precisions 是 p1~p4 的百分比
    return {
        "bleu":   bleu.score / 100,
        "bleu-1": bleu.precisions[0] / 100,
        "bleu-2": bleu.precisions[1] / 100,
        "bleu-3": bleu.precisions[2] / 100,
        "bleu-4": bleu.precisions[3] / 100,
    }

# =========================
# 6. TrainingArguments & Trainer
# =========================
training_args = Seq2SeqTrainingArguments(
    output_dir="./outputs",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="bleu-4",      # 以 BLEU-4 最高时的模型为最优
    greater_is_better=True,

    predict_with_generate=True,          # 让 Trainer 调用 model.generate
    generation_num_beams=5,
    generation_max_length=max_target_len,

    fp16=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collate_fn,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
print("------------------------------------------------------------")
# =========================
# 7. 训练 & 验证
# =========================
trainer.train()
metrics = trainer.evaluate()
print("[FINAL EVAL]", metrics)

# =========================
# 8. 保存模型 & 推理函数
# =========================
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")

# def generate_answer(context: str, question: str, max_len: int = 64) -> str:
#     prompt = f"question: {question}  context: {context}"
#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=max_input_len
#     ).to(device)

#     outputs = model.generate(
#         inputs.input_ids,
#         attention_mask=inputs.attention_mask,
#         max_length=max_len,
#         num_beams=5,
#         early_stopping=True,
#     )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# if __name__ == "__main__":
#     ctx = "淘宝网每年12月31日24:00会对符合条件的扣分做清零处理"
#     q   = "淘宝扣分什么时候清零"
#     print("生成答案：", generate_answer(ctx, q))
