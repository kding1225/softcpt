import os
import sys
import argparse
import torch

from clip.simple_tokenizer import SimpleTokenizer
from clip import clip

def load_clip_to_cpu(backbone_name="RN50"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def get_names(dataset):
    
    if dataset == "general10":
        names = ["caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101"]
    elif dataset == "plant6":
        names = ["fruit_vegetable", "kaggle_flower", "kaggle_mushroom", "kaggle_vegetable", "plant_seedling", "plant_village"]
    elif dataset == "fashion20":
        names = ["pants_type", "pants_length", "waist_type", "collar_type", "sleeve_type", "sleeve_length", "top_pattern", 
          "shoe_material", "shoe_style", "heel_shape", "heel_thickness", "heel_height", "upper_height", "toe_cap_style",
         "hat_style", "socks_length", "socks_type", "skirt_length", "number_of_button_rows", "underwear_style"]
    else:
        raise
    
    return names

parser = argparse.ArgumentParser()
parser.add_argument("ctx", type=str, help="Path to the learned ctx")
parser.add_argument("data", type=str, help="data name")
parser.add_argument("topk", type=int, help="Select top-k similar words")
args = parser.parse_args()

ctx_path = args.ctx
topk = args.topk
out_file = ctx_path + ".words"
if os.path.exists(out_file):
    os.remove(out_file)

assert os.path.exists(ctx_path)
print(f"Return the top-{topk} matched words")

tokenizer = SimpleTokenizer()
clip_model = load_clip_to_cpu()
token_embedding = clip_model.token_embedding.weight
print(f"Size of token embedding: {token_embedding.shape}")

data_names = get_names(args.data)
print(data_names)

ctxs = torch.load(ctx_path, map_location="cpu")
ctxs = ctxs.float()
print(f"Size of context: {ctxs.shape}")

assert len(data_names) == len(ctxs)

for idx, (ctx, name) in enumerate(zip(ctxs, data_names)):
    
    print(f"{idx}/{name}: ")
    line = f"{idx}/{name}: \n"
    
    if ctx.dim() == 2:
        # Generic context
        distance = torch.cdist(ctx, token_embedding)
        print(f"Size of distance matrix: {distance.shape}")
        line += f"Size of distance matrix: {distance.shape}\n"
        sorted_idxs = torch.argsort(distance, dim=1)
        sorted_idxs = sorted_idxs[:, :topk]

        for m, idxs in enumerate(sorted_idxs):
            words = [tokenizer.decoder[idx.item()] for idx in idxs]
            dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
            print(f"{m+1}: {words} {dist}")
            line += f"{m+1}: {words} {dist}\n"
    
    elif ctx.dim() == 3:
        # Class-specific context
        raise NotImplementedError
        
    print()
    line += "\n"
    with open(out_file, "a")  as f:
        f.write(line)
