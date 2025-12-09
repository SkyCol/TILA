import json
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from loader import FewShotImageFolder
from TILA import *

device = "cuda"

## In case you need a VPN:
import os
os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'

# Download the model and config files
hf_hub_download(
    repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    filename="open_clip_pytorch_model.bin",
    local_dir="checkpoints"
)
hf_hub_download(
    repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    filename="open_clip_config.json",
    local_dir="checkpoints"
)

# Load the model and config files 
model_name = "biomedclip_local"
with open("checkpoints/open_clip_config.json", "r") as f:
    config = json.load(f)
    model_cfg = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]

if (not model_name.startswith(HF_HUB_PREFIX)
    and model_name not in _MODEL_CONFIGS
    and config is not None):
    _MODEL_CONFIGS[model_name] = model_cfg

tokenizer = get_tokenizer(model_name) 
model, _, preprocess = create_model_and_transforms(
    model_name=model_name,
    pretrained="checkpoints/open_clip_pytorch_model.bin",
    **{f"image_{k}": v for k, v in preprocess_cfg.items()},
)
model = model.to(device)

# from torchvision import transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(p=0.5),           
#     transforms.Lambda(lambda img: img.convert("RGB")),      
#     transforms.ColorJitter(brightness=0.2, contrast=0.2), 
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
#                          std=(0.26862954, 0.26130258, 0.27577711))  
# ])
# test_transform = transforms.Compose([
#     transforms.Resize((224, 224)),       
#     transforms.Lambda(lambda img: img.convert("RGB")),      
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
#                          std=(0.26862954, 0.26130258, 0.27577711))  
# ])

dataset_root = r'D:\MedicalVLMs\data\Chest X-Ray Images (Pneumonia)' ## Your dataset root
## Folder structure: {dataset_root}/{class_name}/images.jpgs
seed = 2025
fewshot_dataset = FewShotImageFolder(root=dataset_root, num_shots=16, transform=preprocess, test_transform = preprocess, seed=seed)

from torch.utils.data import DataLoader
support_loader = DataLoader(fewshot_dataset, batch_size=32, shuffle=True)

test_dataset = fewshot_dataset.get_test_set()
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Support size: {len(fewshot_dataset)}")
print(f"Test size: {len(test_dataset)}")

idx_to_folder = {v: k for k, v in fewshot_dataset.class_to_idx.items()}
class_names = [folder_name for idx, folder_name in sorted(idx_to_folder.items())]
print(class_names)
num_classes = len(class_names)

prompt_templates = [
    "a photo of {}",
    "a chest X-ray showing {}",
    "a photo of chest X-ray showing {}",
    "radiographic signs of {}",
    "evidence of {} in lungs",
    "an X-ray of lungs with {}",
    "chest radiograph indicating {}",
    "X-ray scan showing {}",
    "frontal chest X-ray with {}",
    "PA chest X-ray of {}",
    "lateral chest X-ray of {}",
    "radiological appearance of {}",
    "findings consistent with {} in chest X-ray",
    "X-ray showing features of {}",
    "lungs affected by {} in X-ray",
    "an X-ray demonstrating {}",
    "medical imaging revealing {}",
    "an image of chest radiograph showing {}",
    "a patient chest X-ray showing {}",
    "radiograph evidence of {}",
    "radiology scan depicting {}",
    "diagnostic chest X-ray of {}",
    "projection X-ray showing {}",
    "X-ray report consistent with {}",
    "lungs diagnosed with {} via X-ray"
]

prompts = [template.format(l) for l in class_names for template in prompt_templates]
tokens = tokenizer(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(tokens)
    text_features = text_features.view(len(class_names), len(prompt_templates), -1)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_feats_global = text_features.mean(dim=1)  # prompt ensemble
    
orig_logit_scale = model.logit_scale.exp()
equiv_logit_scale = compute_equivalent_logit_scale(model, support_loader, text_feats_global, device, orig_logit_scale=orig_logit_scale) 

beta_ot = 10             
layers = [3,5,7,8,9,10,11]
num_layer_to_use = 4

# ------------------ Step 1: Build support prototypes ------------------
layer_scores = compute_layer_scores(model, support_loader, layers, device)
best_layers = select_best_layers(layer_scores, top_k=num_layer_to_use, final_layer=11)
weights = compute_adaptive_layer_weights(layer_scores, best_layers)

cache_keys_layers, cache_values_layers, support_feats_layers, \
cache_keys_layers_proj, cache_values_layers_proj, support_feats_layers_proj = build_class_wise_prototypes(
    model, support_loader, best_layers, device, return_proj="True"
)

# ------------------ Step 2: Build multi-layer adapted text features ------------------
prompts = [template.format(c) for c in class_names for template in prompt_templates]
tokens = tokenizer(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(tokens)
    text_features = text_features.view(len(class_names), len(prompt_templates), -1)
    text_features = F.normalize(text_features, dim=-1)

text_feats_global = build_text_prototypes(text_features, cache_keys_layers_proj, beta=20.0, weights=weights)

# ------------------ Step 3: Compute support alpha ------------------
alpha = compute_robust_support_alpha(support_feats_layers, smooth=False, alpha_min=0.1, alpha_max=0.5)
# print(f"Support alpha: {alpha:.4f}")

# ------------------ Step 4: Infer on test set ------------------
all_labels, all_preds = infer_fewshot(
    model, test_loader, 
    cache_keys_layers, cache_values_layers, text_feats_global, 
    equiv_logit_scale=equiv_logit_scale, a=alpha, beta_ot=10, device=device, layers=best_layers
)

# ------------------ Step 5: Compute metrics ------------------
from sklearn.metrics import balanced_accuracy_score, f1_score
acc = balanced_accuracy_score(all_labels, all_preds)
f1_macro = f1_score(all_labels, all_preds, average='macro')
print(f"Test Accuracy: {acc:.4f}, F1-macro: {f1_macro:.4f}")

## Your will get:
## Test Accuracy: 0.9281, F1-macro: 0.8953
## For CXRP [cell 2018]