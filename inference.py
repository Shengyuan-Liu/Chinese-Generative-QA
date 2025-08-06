#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model(model_dir: str, device: str = None):
    """
    从本地目录加载 tokenizer 和模型，并搬到指定设备。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model, device

def generate_answer(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: str,
    question: str,
    context: str,
    max_input_len: int = 512,
    max_target_len: int = 64,
    num_beams: int = 5,
) -> str:
    """
    给定 question 和 context，返回生成的答案字符串。
    """
    prompt = f"question: {question}  context: {context}"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=max_input_len,
    ).to(device)

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_target_len,
        num_beams=num_beams,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # 1. 指定你要加载的模型目录
    #    - 如果你想用训练中间的 checkpoint，就改成 "./outputs/checkpoint-5000"
    #    - 如果想用最终模型，就用 "./final_model"
    model_dir = "./final_model"

    # 2. 加载
    tokenizer, model, device = load_model(model_dir)

    # 3. 测试生成
    ctx = "我是李明，我的爸爸是李安，我的妈妈是王二。"
    q   = "我的爸爸是谁？"
    ans = generate_answer(tokenizer, model, device, q, ctx)
    print("生成答案：", ans)
