#!/usr/bin/env python
# coding: utf-8

"""
LLaMA and Llama-2 in TransformerLens

这个脚本用于设置环境，加载LLaMA和Llama-2模型，并使用TransformerLens进行模型分析。
"""

import os
import sys
import subprocess
import torch
import tqdm.auto as tqdm
import plotly.io as pio
import plotly.express as px
import pandas as pd
import json

# ===========================
# 安装必要的依赖包
# ===========================

def install_packages():
    """
    安装脚本所需的Python包。
    注意：在生产环境中，建议使用requirements.txt进行依赖管理。
    """
    required_packages = [
        "transformers==4.31.0",
        "sentencepiece",
        "transformer_lens",
        "circuitsvis",
        "bitsandbytes==0.42.0",
        "accelerate",
        "plotly",
        "tqdm",
        "jaxtyping",
        "torch"  # 确保根据你的系统和需求安装适当版本的torch
    ]

    for package in required_packages:
        try:
            __import__(package.split("==")[0])
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 如果需要，取消注释以下行以自动安装缺失的包
# install_packages()

# ===========================
# 环境设置
# ===========================

DEVELOPMENT_MODE = False
IN_VSCODE = False
IN_GITHUB = os.getenv("GITHUB_ACTIONS") == "true"

try:
    import google.colab
    IN_COLAB = True
    print("Running as a Colab notebook")
except ImportError:
    IN_COLAB = False
    print("Running as a local script or Jupyter notebook")
    # 自动重新加载模块以便于开发
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython:
            ipython.magic("load_ext autoreload")
            ipython.magic("autoreload 2")
    except ImportError:
        pass  # 不是在IPython环境中运行

# 设置Plotly渲染器
if IN_COLAB or not DEVELOPMENT_MODE:
    pio.renderers.default = "colab"
elif IN_VSCODE:
    pio.renderers.default = "notebook_connected"
else:
    pio.renderers.default = "browser"  # 默认使用浏览器渲染
print(f"Using Plotly renderer: {pio.renderers.default}")

# ===========================
# 导入其他模块
# ===========================

import circuitsvis as cv
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from tqdm import tqdm
from jaxtyping import Float
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer

# 禁用梯度计算
torch.set_grad_enabled(False)

# ===========================
# 定义辅助函数
# ===========================

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    """
    显示图像张量。
    """
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        **kwargs
    ).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    """
    绘制折线图。
    """
    px.line(
        utils.to_numpy(tensor),
        labels={"x": xaxis, "y": yaxis},
        **kwargs
    ).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    """
    绘制散点图。
    """
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        x=x,
        y=y,
        labels={"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs
    ).show(renderer)

# ===========================
# 加载LLaMA模型
# ===========================

def load_llama_model(model_path):
    """
    加载LLaMA模型。

    参数:
        model_path (str): LLaMA模型的路径。

    返回:
        model: HookedTransformer模型实例。
        tokenizer: LLaMA的tokenizer。
    """
    if not model_path:
        print("请设置MODEL_PATH变量，指向转换后的LLaMA权重目录。")
        return None, None

    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    hf_model = LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)

    model = HookedTransformer.from_pretrained(
        "llama-7b",
        hf_model=hf_model,
        device="cpu",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=tokenizer
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"模型已加载到 {device} 设备上。")

    # 示例生成
    output = model.generate("The capital of Germany is", max_new_tokens=20, temperature=0)
    print(output)

    return model, tokenizer

# ===========================
# 加载LLaMA-2量化模型
# ===========================

def load_llama2_quantized_model(model_path, device_map="cuda:0", load_in_4bit=True):
    """
    加载量化的LLaMA-2模型。

    参数:
        model_path (str): LLaMA-2模型在HuggingFace上的路径。
        device_map (str): 设备映射，默认使用CUDA:0。
        load_in_4bit (bool): 是否以4位量化加载模型。

    返回:
        model: HookedTransformer模型实例。
        tokenizer: LLaMA-2的tokenizer。
        hf_model: 原生的HuggingFace模型。
    """
    inference_dtype = torch.float32
    # inference_dtype = torch.float16  # 根据需求调整

    print("加载量化的LLaMA-2模型...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=inference_dtype,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(load_in_4bit=load_in_4bit)
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = HookedTransformer.from_pretrained(
        model_path,
        hf_model=hf_model,
        dtype=inference_dtype,
        fold_ln=False,
        fold_value_biases=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=tokenizer
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"量化模型已加载到 {device} 设备上。")

    # 示例生成
    output = model.generate("The capital of Germany is", max_new_tokens=2, temperature=0)
    print(output)

    return model, tokenizer, hf_model

# ===========================
# 验证GPU内存使用
# ===========================

def verify_gpu_memory():
    """
    打印GPU的可用内存和总内存（单位：GB）。
    """
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        print(f"GPU Memory - Free: {free_mem / 1e9:.2f} GB, Total: {total_mem / 1e9:.2f} GB")
    else:
        print("CUDA不可用，无法获取GPU内存信息。")

# ===========================
# 比较HuggingFace模型的logits
# ===========================

def compare_logits(prompts, model, hf_model, tokenizer):
    """
    比较TransformerLens模型和HuggingFace模型的logits。

    参数:
        prompts (list): 输入的提示语列表。
        model: TransformerLens的模型实例。
        hf_model: HuggingFace的模型实例。
        tokenizer: 相应的tokenizer。
    """
    model.eval()
    hf_model.eval()

    prompt_ids = [tokenizer.encode(prompt, return_tensors="pt").to(model.device) for prompt in prompts]
    tl_logits = [model(prompt_ids_i).detach().cpu() for prompt_ids_i in tqdm.tqdm(prompt_ids, desc="计算TransformerLens logits")]

    # 将hf_model移动到相同的设备以加速计算
    hf_model.to(model.device)
    logits = [hf_model(prompt_ids_i).logits.detach().cpu() for prompt_ids_i in tqdm.tqdm(prompt_ids, desc="计算HuggingFace logits")]

    for i in range(len(prompts)):
        if i == 0:
            print(f"logits[{i}] dtype: {logits[i].dtype}, values: {logits[i]}")
            print(f"tl_logits[{i}] dtype: {tl_logits[i].dtype}, values: {tl_logits[i]}")
        try:
            assert torch.allclose(logits[i], tl_logits[i], atol=1e-4, rtol=1e-2)
            print(f"Prompt {i} 的logits比较通过。")
        except AssertionError:
            print(f"Prompt {i} 的logits比较失败。")

# ===========================
# TransformerLens 演示
# ===========================

def transformerlens_demo(model, tokenizer):
    """
    使用TransformerLens进行模型分析的演示。

    参数:
        model: TransformerLens的模型实例。
        tokenizer: 相应的tokenizer。
    """
    # 读取模型的注意力模式
    llama_text = (
        "Natural language processing tasks, such as question answering, machine translation, "
        "reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."
    )
    llama_tokens = model.to_tokens(llama_text)
    llama_logits, llama_cache = model.run_with_cache(llama_tokens, remove_batch_dim=True)

    attention_pattern = llama_cache["pattern", 0, "attn"]
    llama_str_tokens = model.to_str_tokens(llama_text)

    print("Layer 0 Head Attention Patterns:")
    cv.attention.attention_patterns(tokens=llama_str_tokens, attention=attention_pattern)

    # 定义一个头部消融的hook
    layer_to_ablate = 0
    head_index_to_ablate = 31

    def head_ablation_hook(
        value: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        print(f"Shape of the value tensor: {value.shape}")
        value[:, :, head_index_to_ablate, :] = 0.0
        return value

    # 计算原始损失
    original_loss = model(llama_tokens, return_type="loss")

    # 计算消融后的损失
    ablated_loss = model.run_with_hooks(
        llama_tokens,
        return_type="loss",
        fwd_hooks=[(
            utils.get_act_name("v", layer_to_ablate),
            head_ablation_hook
        )]
    )

    print(f"Original Loss: {original_loss.item():.3f}")
    print(f"Ablated Loss: {ablated_loss.item():.3f}")

# ===========================
# 主函数
# ===========================

def main():
    """
    主函数，执行脚本的主要逻辑。
    """
    # 设置MODEL_PATH，指向转换后的LLaMA权重目录
    MODEL_PATH = '/aifs4su/yaodong/changye/final.json'  # 根据实际情况修改

    # 加载LLaMA模型
    model, tokenizer = load_llava_model(MODEL_PATH)

    if not model:
        print("LLaMA模型未加载，请检查MODEL_PATH。")
        return

    # 加载量化的LLaMA-2模型
    LLAMA_2_7B_CHAT_PATH = "meta-llama/Llama-2-7b-chat-hf"  # 修改为实际路径
    llama2_model, llama2_tokenizer, hf_model = load_llama2_quantized_model(LLAMA_2_7B_CHAT_PATH)

    # 验证GPU内存使用
    verify_gpu_memory()

    # 比较logits
    prompts = [
        "The capital of Germany is",
        "2 * 42 = ",
        "My favorite",
        "aosetuhaosuh aostud aoestuaoentsudhasuh aos tasat naostutshaosuhtnaoe usaho uaotsnhuaosntuhaosntu haouaoshat u saotheu saonuh aoesntuhaosut aosu thaosu thaoustaho usaothusaothuao sutao sutaotduaoetudet uaosthuao uaostuaoeu aostouhsaonh aosnthuaoscnuhaoshkbaoesnit haosuhaoe uasotehusntaosn.p.uo ksoentudhao ustahoeuaso usant.hsa otuhaotsi aostuhs",
    ]
    compare_logits(prompts, llama2_model, hf_model, llama2_tokenizer)

    # TransformerLens 演示
    transformerlens_demo(llama2_model, llama2_tokenizer)

if __name__ == "__main__":
    main()
