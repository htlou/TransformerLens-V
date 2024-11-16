import sys
from tqdm import tqdm
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
# sys.path.append('/aifs4su/yaodong/changye/TransformerLens_soft')
from transformer_lens.HookedLlava import HookedLlava
import pdb
pdb.set_trace()
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_PATH="/home/saev/changye/model/Mistral-7B-Instruct-v0.2"
def load_models_and_processor(model_name,model_path):
    """
    加载处理器、视觉-语言模型和HookedTransformer语言模型。
    """
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hook_language_model = HookedLlava.from_pretrained(
        model_name,
        hf_model=hf_model,
        device="cuda", 
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=None,
        dtype=torch.float32,
        vision_tower=None,
        multi_modal_projector=None,
        
    )
    # print(hook_language_model.state_dict().keys())
    # print(vision_model.language_model.state_dict().keys())
    # 将模型转移到GPU（如果可用）
    hook_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hf_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    hook_language_model = hook_language_model.to(hook_device)
    hf_model = hf_model.to(hf_device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return hf_model, hook_language_model, tokenizer

def consistent_check(model, hf_model, tokenizer):
    """
    检查 HookedTransformer 模型输出与 Hugging Face 模型输出的一致性。
    避免内存不够时的情况，逐步处理每个输入。
    """
    prompts = [
        "The capital of Germany is",
        "2 * 42 = ", 
        "My favorite", 
        "aosetuhaosuh aostud aoestuaoentsudhasuh aos tasat naostutshaosuhtnaoe usaho uaotsnhuaosntuhaosntu haouaoshat u saotheu saonuh aoesntuhaosut aosu thaosu thaoustaho usaothusaothuao sutao sutaotduaoetudet uaosthuao uaostuaoeu aostouhsaonh aosnthuaoscnuhaoshkbaoesnit haosuhaoe uasotehusntaosn.p.uo ksoentudhao ustahoeuaso usant.hsa otuhaotsi aostuhs",
    ]
    
    # 切换到评估模式
    model.eval()
    hf_model.eval()

    # 将模型参数移动到 GPU 上
    model_device = next(model.parameters()).device
    hf_model_device = next(hf_model.parameters()).device
    
    # 分别处理每一个 prompt，避免一次性加载太多
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")

        # 将输入移动到模型所在的设备
        prompt_id = tokenizer.encode(prompt, return_tensors="pt").to(model_device)
        prompt_id_hf = tokenizer.encode(prompt, return_tensors="pt").to(hf_model_device)

        # 分别计算 HookedTransformer 和 Hugging Face 模型的输出
        tl_logits = model(prompt_id).detach().cpu()
        hf_logits = hf_model(prompt_id_hf).logits.detach().cpu()

        # 比较输出是否在允许的误差范围内
        if not torch.allclose(hf_logits, tl_logits, atol=1e-4, rtol=1e-2):
            print(f"Difference found in prompt {i}:")
            print(f"hf_logits: {hf_logits}")
            print(f"tl_logits: {tl_logits}")
            print(f"Difference: {hf_logits - tl_logits}")
            
            # 打印最大绝对误差和相对误差
            abs_diff = torch.max(torch.abs(hf_logits - tl_logits))
            rel_diff = torch.max(torch.abs((hf_logits - tl_logits) / (tl_logits + 1e-8)))
            print(f"Max absolute difference: {abs_diff.item()}")
            print(f"Max relative difference: {rel_diff.item()}")

            # 放宽误差条件
            if not torch.allclose(hf_logits, tl_logits, atol=1e-3, rtol=1e-2):
                print(f"Larger difference persists for prompt {i}, investigate further.")
        
        # 断言条件，严格验证差异
        assert torch.allclose(hf_logits, tl_logits, atol=1e-4, rtol=1e-2)

    print("Consistency check completed.")
    
def main():
    # 加载模型和处理器
    hf_model, hook_language_model, tokenizer = load_models_and_processor(MODEL_NAME,MODEL_PATH)
    
    # 进行一致性检查
    #目前hf_model和hook_language_model的attention不同，hf采用SdpaAttention而hook_language_model采用普通的Attention，所以会有一些误差
    #但又由于llavamistral采用了普通的Attention，所以这里的sae可能还要考虑一下选用哪一种Attention下的模型（1022）
    consistent_check(hook_language_model, hf_model, tokenizer)

    inputs="The capital of America is"
    inputs=tokenizer(inputs,return_tensors="pt")["input_ids"]
    inputs=inputs.to("cuda:0")
    outputs = hook_language_model.generate(inputs)
    # outputs = hf_model.generate(inputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 打印生成的文本
    print(generated_text)
if __name__ == "__main__":
    main()