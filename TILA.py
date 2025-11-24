# tila_clip.py
import torch
import torch.nn.functional as F
from collections import defaultdict

# ------------------ Robust Prototype Class ------------------
class RobustPrototype:
    def __init__(self, num_classes, feat_dim, beta_density=10.0, device="cuda"):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.beta_density = beta_density
        self.device = device
        self.prototypes = torch.zeros(num_classes, feat_dim, device=device)

    def compute_mahalanobis_weighted(self, feats_c):
        feats_c = F.normalize(feats_c, dim=-1)
        if feats_c.shape[0] == 1:
            return feats_c.squeeze(0)
        mean_c = feats_c.mean(dim=0, keepdim=True)
        cov_c = torch.cov(feats_c.T) + 1e-6 * torch.eye(feats_c.shape[1], device=feats_c.device)
        inv_cov = torch.linalg.pinv(cov_c)
        diff = feats_c - mean_c
        mahal = torch.sqrt(torch.sum((diff @ inv_cov) * diff, dim=1))
        w_mahal = torch.softmax(-mahal, dim=0)
        return (w_mahal.unsqueeze(1) * feats_c).sum(dim=0)

    def init_prototypes(self, support_feats):
        for c, feats_c in support_feats.items():
            feats_c = F.normalize(feats_c, dim=-1)
            proto_mahal = self.compute_mahalanobis_weighted(feats_c)
            self.prototypes[c] = F.normalize(proto_mahal, dim=-1)

    def get_prototypes(self):
        return self.prototypes


# ------------------ Feature Extraction ------------------
def extract_image_features(model, images, layers=[5,7,9,11]):
    features = {}
    hooks = []

    def get_hook(idx):
        def hook(module, input, output):
            features[idx] = output[:, 0]  # CLS token
        return hook

    trunk = model.visual.trunk
    for idx in layers:
        hooks.append(trunk.blocks[idx].register_forward_hook(get_hook(idx)))

    _ = model.encode_image(images)

    for h in hooks:
        h.remove()

    return [features[idx] for idx in layers]


# ------------------ Build multi-layer adapted text features ------------------
def build_multilayer_adapted_text_features(text_features, cache_keys_layers, beta=20.0, weights=None):
    num_classes, num_prompts, feat_dim = text_features.shape
    num_layers = len(cache_keys_layers)

    if weights is None:
        weights = [1.0 / num_layers] * num_layers

    adapted_text_feats = torch.zeros(num_classes, feat_dim, device=text_features.device)

    for c in range(num_classes):
        feats_c = F.normalize(text_features[c], dim=-1)
        layer_weighted = []

        for l, rp in enumerate(cache_keys_layers):
            proto_c = rp.get_prototypes()[c].unsqueeze(0)
            sim = (feats_c @ proto_c.T).squeeze(-1)
            w = torch.softmax(beta * sim, dim=0)
            proto_text = (w.unsqueeze(1) * feats_c).sum(dim=0)
            layer_weighted.append(weights[l] * proto_text)

        fused_proto = torch.stack(layer_weighted, dim=0).sum(dim=0)
        adapted_text_feats[c] = F.normalize(fused_proto, dim=-1)

    return adapted_text_feats


# ------------------ Build prototypes from support set ------------------
def build_support_prototypes(model, train_loader, device="cuda", layers=[5,7,9,11]):
    cache_keys_layers = []
    cache_values_layers = []

    with torch.no_grad():
        for layer in layers:
            class_feats = defaultdict(list)
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                feats = extract_image_features(model, images, layers=[layer])[0]
                feats = F.normalize(feats, dim=-1)
                for i, label in enumerate(labels):
                    class_feats[label.item()].append(feats[i])

            class_ids = sorted(class_feats.keys())
            num_classes = len(class_ids)
            support_feats = {c: torch.stack(v, dim=0).to(device) for c, v in class_feats.items()}

            rp = RobustPrototype(num_classes, feats.shape[1], device=device)
            rp.init_prototypes(support_feats)
            cache_keys_layers.append(rp)

            values = [F.one_hot(torch.tensor(class_ids.index(c)), num_classes=num_classes).float() for c in class_ids]
            cache_values_layers.append(torch.stack(values).to(device))

    return cache_keys_layers, cache_values_layers, class_ids


# ------------------ Inference using cached prototypes ------------------
def infer_fewshot(model, test_loader, cache_keys_layers, cache_values_layers, text_feats_global, equiv_logit_scale=40, a=0.5, beta_ot=10, device="cuda", layers=[5,7,9,11]):
    weights = torch.tensor([1.0/len(layers)]*len(layers)).to(device)
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            layer_feats = extract_image_features(model, images, layers=layers)

            layer_logits = []
            for idx, feats in enumerate(layer_feats):
                feats = F.normalize(feats, dim=-1)
                proto_layer = cache_keys_layers[idx].get_prototypes()
                affinity = feats @ proto_layer.T
                cache_logits = ((-1)*(beta_ot - beta_ot*affinity)).exp() @ cache_values_layers[idx]
                layer_logits.append(cache_logits)

            tip_logits = sum(w*l for w,l in zip(weights, layer_logits))

            clip_feats = model.encode_image(images)
            clip_feats = F.normalize(clip_feats, dim=-1)
            clip_logits = clip_feats @ text_feats_global.T

            tip_logits = equiv_logit_scale * (a*clip_logits + (1-a)*tip_logits)/2

            # Z-score norm
            clip_mean, clip_std = clip_logits.mean(1, keepdim=True), clip_logits.std(1, keepdim=True)+1e-6
            tip_mean, tip_std = tip_logits.mean(1, keepdim=True), tip_logits.std(1, keepdim=True)+1e-6
            tip_logits = ((tip_logits-tip_mean)/tip_std)*clip_std + clip_mean

            preds = tip_logits.softmax(1).argmax(1)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    return all_labels, all_preds

# import json
# from urllib.request import urlopen
# from PIL import Image
# import torch
# from huggingface_hub import hf_hub_download
# from open_clip import create_model_and_transforms, get_tokenizer
# from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS


# # Download the model and config files
# hf_hub_download(
#     repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
#     filename="open_clip_pytorch_model.bin",
#     local_dir="checkpoints"
# )
# hf_hub_download(
#     repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
#     filename="open_clip_config.json",
#     local_dir="checkpoints"
# )


# # Load the model and config files 
# model_name = "biomedclip_local"

# with open("checkpoints/open_clip_config.json", "r") as f:
#     config = json.load(f)
#     model_cfg = config["model_cfg"]
#     preprocess_cfg = config["preprocess_cfg"]


# if (not model_name.startswith(HF_HUB_PREFIX)
#     and model_name not in _MODEL_CONFIGS
#     and config is not None):
#     _MODEL_CONFIGS[model_name] = model_cfg

# tokenizer = get_tokenizer(model_name)

# model, _, preprocess = create_model_and_transforms(
#     model_name=model_name,
#     pretrained="checkpoints/open_clip_pytorch_model.bin",
#     **{f"image_{k}": v for k, v in preprocess_cfg.items()},
# )

# model.to(device)
# prompt_templates = [
#         "a photo of chest X-ray showing {}",
#         "a chest X-ray showing {}.",
#         "evidence of {} in lungs",
#         "radiographic signs of {}.",
#         "a patient diagnosed with {} in chest X-ray."
#     ]
# prompts = [template.format(l) for l in class_names for template in prompt_templates]
# tokens = tokenizer(prompts).to(device)
# with torch.no_grad():
#     text_features = model.encode_text(tokens)
#     text_features = text_features.view(len(class_names), len(prompt_templates), -1)
#     text_features /= text_features.norm(dim=-1, keepdim=True)
#     text_feats_global = text_features.mean(dim=1)  # prompt ensemble


## Example: ## See toturials.ipynb

# # ------------------ Compute inter-class affinites from support set ------------------
# cache_keys_layers, cache_values_layers, class_ids = build_support_prototypes(model, train_loader, device=device) 

# # ------------------ Text feature weighting: Compute Adapted Text Features ------------------
# with torch.no_grad():
#     tokens = tokenizer(prompts).to(device)
#     text_features = model.encode_text(tokens)
#     text_features = text_features.view(len(class_names), len(prompt_templates), -1)
#     text_features = F.normalize(text_features, dim=-1)

#     # Now update text features using multi-layer robust prototypes
#     text_feats_global = build_multilayer_adapted_text_features(
#         text_features, cache_keys_layers, beta=20.0,
#         weights=[0.25, 0.25, 0.25, 0.25]   # same as used in fusion
#     )


# # Inference
# all_labels, all_preds = infer_fewshot(model, test_loader, cache_keys_layers, cache_values_layers, text_feats_global)
