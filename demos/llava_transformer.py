import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, LlavaNextImageProcessor, AutoTokenizer
from transformer_lens import HookedTransformer
from PIL import Image

MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"

if MODEL_PATH:
    # 加载处理器和视觉-语言模型
    processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)
    vision_model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    print("Vision model loaded.")

    # 加载 HookedTransformer 语言模型
    hf_model = vision_model.language_model
    model = HookedTransformer.from_pretrained(
        MODEL_PATH,
        hf_model=hf_model,
        device="cpu",  # Load on CPU initially
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=None,
    )
    
    # 将模型转移到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    vision_model = vision_model.to(device)

    # 加载本地图像
    image_path = "/data/changye/IMG_20190131_173653.jpg"
    image = Image.open(image_path)

    # 使用处理器处理图像和文本
    chat_template = "[INST] <image>\nWhat is shown in this image? [/INST]"
    inputs = processor(images=image, text=chat_template, return_tensors="pt").to(device)

    # 提取视觉特征
    pixel_values = inputs['pixel_values']  # 获取图像的pixel_values
    with torch.no_grad():
        vision_features = vision_model.vision_tower(pixel_values)  # 提取视觉特征

    # 提取文本的 input_ids
    input_ids = inputs['input_ids']

    #这一部分需要调研llava是如何拼接的
    # # 将视觉特征和文本 input_ids 结合
    # # 将视觉特征与文本的 token 特征联合输入 HookedTransformer 模型
    # # 需要确保维度匹配
    # combined_input = torch.cat([vision_features, input_ids], dim=-1)  # 示例：将视觉特征和文本结合

    # # 调用模型的 generate 函数，假设它支持联合输入
    # output_ids = model.generate(input=combined_input, max_new_tokens=100, temperature=0)

    # # 解码并打印生成的文本输出
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # print(output_text)
