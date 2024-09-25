from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import pdb
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float32, low_cpu_mem_usage=True) 
model.to("cuda:0")
# 打印模型的配置
print(model.config)

# 检查是否存在注意力类型的字段 (例如对于某些模型，它可能是 attn_types)
if hasattr(model.config, "attn_types"):
    print(f"模型的注意力类型: {model.config.attn_types}")
else:
    print("模型配置中没有显式的注意力类型字段")

# /aifs4su/yaodong/changye/TransformerLens/IMG_20230213_181559.jpg
image=Image.open("/aifs4su/yaodong/changye/IMG_20200201_111602.jpg")
# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What is shown in this image?"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

# autoregressively complete prompt
# pdb.set_trace()
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))
