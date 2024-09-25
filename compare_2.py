import sys
from tqdm import tqdm
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    AutoModelForCausalLM,
)

sys.path.append('/aifs4su/yaodong/changye/TransformerLens')
from transformer_lens.HookedLlava import HookedLlava
import pdb
pdb.set_trace()
MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"

def load_models_and_processor(model_path):
    """
    加载处理器、视觉-语言模型和HookedTransformer语言模型。
    """
    # 加载处理器和视觉-语言模型
    processor = LlavaNextProcessor.from_pretrained(model_path)
    vision_model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True
    )
    print("Vision model loaded.")
    
    # 加载 HookedTransformer 语言模型
    hook_language_model = HookedLlava.from_pretrained(
        model_path,
        hf_model=vision_model.language_model,
        device="cuda:4", 
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=None,
        dtype=torch.float32,
    )

    # 将模型转移到GPU（如果可用）
    hook_device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    hook_language_model = hook_language_model.to(hook_device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return processor, vision_model, hook_language_model, tokenizer

def generate_text_with_hooked_model(prompt, hook_language_model, tokenizer):
    """
    使用HookedTransformer语言模型生成文本。
    """
    # 将prompt转换为token


    # 生成文本
    with torch.no_grad():
        generated_ids = hook_language_model.generate(input=prompt)

    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

def generate_text_with_vision_model(prompt, vision_model, tokenizer):
    """
    使用视觉-语言模型生成文本。
    """
    # 将prompt转换为token
    inputs = tokenizer(prompt, return_tensors="pt").to(vision_model.device)

    # 模型生成输出
    with torch.no_grad():
        generated_ids = vision_model.generate(**inputs, max_length=50)

    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# 加载模型和处理器
processor, vision_model, hook_language_model, tokenizer = load_models_and_processor(MODEL_PATH)

# 定义要测试的文本
prompt = "The capital of Germany is"

# 使用 HookedTransformer 语言模型生成文本
hooked_generated_text = generate_text_with_hooked_model(prompt, hook_language_model, tokenizer)
print("Generated Text using HookedTransformer:", hooked_generated_text)


