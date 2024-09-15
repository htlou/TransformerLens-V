import torch
import tqdm.auto as tqdm
import plotly.express as px

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from jaxtyping import Float
import sys
sys.path.append('/aifs4su/yaodong/changye/TransformerLens')
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)
    

import pdb; 
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
hf_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

model = HookedTransformer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device="cuda", fold_ln=False, center_writing_weights=False, center_unembed=False)
model.eval()
hf_model.eval()
prompts = [
    "The capital of Germany is",
    "2 * 42 = ", 
    "My favorite", 
    "aosetuhaosuh aostud aoestuaoentsudhasuh aos tasat naostutshaosuhtnaoe usaho uaotsnhuaosntuhaosntu haouaoshat u saotheu saonuh aoesntuhaosut aosu thaosu thaoustaho usaothusaothuao sutao sutaotduaoetudet uaosthuao uaostuaoeu aostouhsaonh aosnthuaoscnuhaoshkbaoesnit haosuhaoe uasotehusntaosn.p.uo ksoentudhao ustahoeuaso usant.hsa otuhaotsi aostuhs",
]

model.eval()
hf_model.eval()
prompt_ids = [tokenizer.encode(prompt, return_tensors="pt") for prompt in prompts]
tl_logits = [model(prompt_ids).detach().cpu() for prompt_ids in tqdm(prompt_ids)]

# hf logits are really slow as it's on CPU. If you have a big/multi-GPU machine, run `hf_model = hf_model.to("cuda")` to speed this up
logits = [hf_model(prompt_ids).logits.detach().cpu() for prompt_ids in tqdm(prompt_ids)]

for i in range(len(prompts)): 
    assert torch.allclose(logits[i], tl_logits[i], atol=1e-4, rtol=1e-2)