from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

# 加载处理器和模型
# processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
# model = LlavaNextForConditionalGeneration.from_pretrained(
#     "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True
# )
# model.to("cuda:0")
# llm=model.language_model
# # 访问模型的配置
# config = llm.config
# print(f"d_model: {config.hidden_size}")
# print(f"d_head: {config.hidden_size // config.num_attention_heads}")
# print(f"n_heads: {config.num_attention_heads}")
# print(f"d_mlp: {config.intermediate_size}")
# print(f"n_layers: {config.num_hidden_layers}")
# print(f"n_ctx: {config.max_position_embeddings}")
# print(f"d_vocab: {config.vocab_size}")
# print(f"act_fn: {config.hidden_act}")
# print(f"normalization_type: {getattr(config, 'normalization_type', 'Not available')}")
# print(f"positional_embedding_type: {getattr(config, 'position_embedding_type', 'Not available')}")
# print(f"window_size: {getattr(config, 'window_size', 'Not available')}")
# print(f"attn_types: {getattr(config, 'attn_types', 'Not available')}")
# print(f"eps: {getattr(config, 'layer_norm_eps', 'Not available')}")
# print(f"n_key_value_heads: {getattr(config, 'num_key_value_heads', 'Not available')}")
# print(f"gated_mlp: {getattr(config, 'gated_mlp', 'Not available')}")
# print(f"use_local_attn: {getattr(config, 'use_local_attn', 'Not available')}")
# print(f"rotary_dim: {config.hidden_size // config.num_attention_heads}")

# 你可以根据需求访问更多的配置参数


# 假设已经加载了 VLM 模型，例如 Llava 模型
vlm_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# 提取语言模型部分
language_model = vlm_model.language_model  # Llava 模型中的语言模型部分

# 检查语言模型的类型
print(f"语言模型的类型: {type(language_model)}")

# 检查语言模型的参数数量
param_count = sum(p.numel() for p in language_model.parameters())
print(f"语言模型的参数数量: {param_count}")

# 保存语言模型到文件
language_model.save_pretrained("/aifs4su/yaodong/changye/model/llava")
print("语言模型已保存")
