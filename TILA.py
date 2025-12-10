# tila, inplementation with BiomedCLIP
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

def extract_image_features(model, images, layers):
    """
    Extract the image features from biomedclip image encoder (without projection).
    Keyword arguments:
    argument -- model: loaded model (biomedclip)
    argument -- images: the images to use
    argument -- layers: the layers to use
    Return: feats_out: the image features
    """
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

def extract_image_features_withproj(model, images, layers=[5,7,9,11]):
    """
    Extract the image features from biomedclip image encoder (with projection).
    Keyword arguments:
    argument -- model: loaded model (biomedclip)
    argument -- images: the images to use
    argument -- layers: the layers to use
    Return: feats_out: the image features
    """
    features = {}
    hooks = []

    def get_hook(idx):
        def hook(module, input, output):
            # CLS token
            features[idx] = output[:, 0]
        return hook

    trunk = model.visual.trunk
    for idx in layers:
        hooks.append(trunk.blocks[idx].register_forward_hook(get_hook(idx)))

    _ = model.encode_image(images)  # forward pass

    for h in hooks:
        h.remove()

    # project into the same space as text features
    feats_out = []
    for idx in layers:
        proj = model.visual.head(features[idx])   # projection
        feats_out.append(F.normalize(proj, dim=-1))
    return feats_out


class RobustPrototype:
    """ Build Prototype
    Build class-wise prototype based on the support images.
    Keyword arguments:
    argument -- num_classes: the number of classes in the support set
    argument -- feat_dim: the dimension of the feature
    argument -- beta_density: the beta value for the density
    argument -- momentum: the momentum for the prototype
    argument -- device: the device to use
    Return: 
    """
    
    def __init__(self, num_classes, feat_dim, beta_density=10.0, momentum=0.1, device='cuda'):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.beta_density = beta_density
        self.momentum = momentum
        self.device = device
        self.prototypes = torch.zeros(num_classes, feat_dim, device=device)

    def compute_mahalanobis_weighted(self, feats_c):
        feats_c = F.normalize(feats_c, dim=-1)

        # One-shot case
        if feats_c.shape[0] == 1:
            return feats_c.squeeze(0)  # just return the single normalized feature

        # Few-shot case
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
            # 1. Mean
            proto_mean = feats_c.mean(dim=0)
            # 2. Density (BECLR style)
            sim_mat = feats_c @ feats_c.T
            dens = sim_mat.mean(dim=1)
            w_density = torch.softmax(self.beta_density * dens, dim=0)
            proto_density = (w_density.unsqueeze(1) * feats_c).sum(dim=0)
            # 3. Mahalanobis-weighted
            proto_mahal = self.compute_mahalanobis_weighted(feats_c)
            # Combine all
            # proto = F.normalize(proto_mean + proto_density + proto_mahal, dim=-1)
            # proto = F.normalize(proto_density + proto_mahal, dim=-1)
            proto = F.normalize(proto_mahal, dim=-1) 
            self.prototypes[c] = proto

    def get_prototypes(self):
        return self.prototypes
    
def compute_layer_scores(model, train_loader, layers, device):
    """
    Compute the layer scores from the support set.
    Keyword arguments:
    argument -- model: loaded model (biomedclip)
    argument -- train_loader: the train loader to use (support loader)
    argument -- layers: the layers to use
    argument -- device: the device to use
    Return: layer_scores: the layer scores for the support set
    """
    layer_scores = {}
    with torch.no_grad():
        feats_per_layer = {l: defaultdict(list) for l in layers}
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            layer_feats = extract_image_features(model, images, layers)
            for i, l in enumerate(layers):
                feats = F.normalize(layer_feats[i], dim=-1)
                for j, y in enumerate(labels):
                    feats_per_layer[l][y.item()].append(feats[j])

        for l in layers:
            class_feats = feats_per_layer[l]
            class_ids = sorted(class_feats.keys())
            # The lower the inter-class similarity, the better, the higher the layer score.
            inter_sims = []
            for i in range(len(class_ids)):
                for j in range(i+1, len(class_ids)):
                    proto_i = torch.stack(class_feats[class_ids[i]]).mean(0)
                    proto_j = torch.stack(class_feats[class_ids[j]]).mean(0)
                    inter_sims.append(F.cosine_similarity(proto_i, proto_j, dim=0).item())
            layer_scores[l] = 1 - np.mean(inter_sims)
    return layer_scores

def select_best_layers(layer_scores, top_k=4, final_layer=11):
    """
    Select the best layers from the layer scores.
    Keyword arguments:
    argument -- layer_scores: the layer scores to use (dict, key: layer, value: layer score)
    argument -- top_k: the number of layers to select
    argument -- final_layer: the final layer to use
    Return: selected: the selected layers
    """

    layers_except_final = [(l, s) for l, s in layer_scores.items() if l != final_layer]
    layers_sorted = sorted(layers_except_final, key=lambda x: x[1], reverse=True)
    
    selected = [l for l, _ in layers_sorted[:top_k - 1]]

    # Add the final layer
    selected.append(final_layer)

    return selected

def compute_adaptive_layer_weights(layer_scores, selected_layers, tau=1.0):
    """
    Compute adaptive weights for selected layers including last layer.
    Normalize to sum = 1.
    
    tau > 1: sharper distribution (more emphasis on largest scores)
    tau < 1: softer distribution (more uniform)
    """
    scores = np.array([layer_scores.get(l, 1e-6) for l in selected_layers])
    weights = tau * scores / np.sum(tau * scores)
    return weights.tolist()


def build_class_wise_prototypes(model, train_loader, best_layers, device, return_proj=False):
    """
    Build support prototypes for each selected layer, with optional projection.

    Args:
        model: the model
        train_loader: support set loader
        best_layers: list of layer indices
        device: device
        return_proj: if True, also compute prototypes with projection

    Returns:
        If return_proj=False:
            cache_keys_layers, cache_values_layers, support_feats_layers
        If return_proj=True:
            cache_keys_layers_no, cache_values_layers_no, support_feats_layers_no,
            cache_keys_layers_proj, cache_values_layers_proj, support_feats_layers_proj
    """
    cache_keys_layers_no, cache_values_layers_no, support_feats_layers_no = [], [], []
    cache_keys_layers_proj, cache_values_layers_proj, support_feats_layers_proj = [], [], []

    with torch.no_grad():
        for layer in best_layers:
            # ------------------- Collect support features for this layer -------------------
            class_feats_no = defaultdict(list)
            class_feats_proj = defaultdict(list) if return_proj else None

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                feats_no = extract_image_features(model, images, layers=[layer])[0]
                feats_no = F.normalize(feats_no, dim=-1)

                for i, label in enumerate(labels):
                    class_feats_no[label.item()].append(feats_no[i])

                if return_proj:
                    feats_proj = extract_image_features_withproj(model, images, layers=[layer])[0]
                    feats_proj = F.normalize(feats_proj, dim=-1)
                    for i, label in enumerate(labels):
                        class_feats_proj[label.item()].append(feats_proj[i])

            # ------------------- Convert lists to tensors -------------------
            class_ids = sorted(class_feats_no.keys())

            support_feats_no = {c: torch.stack(v, dim=0).to(device) for c, v in class_feats_no.items()}
            support_feats_layers_no.append(support_feats_no)

            num_classes = len(class_ids)
            feat_dim_no = feats_no.shape[1]

            rp_no = RobustPrototype(num_classes, feat_dim_no, device=device)
            rp_no.init_prototypes(support_feats_no)
            cache_keys_layers_no.append(rp_no)

            values_no = [F.one_hot(torch.tensor(class_ids.index(c)), num_classes=num_classes).float()
                         for c in class_ids]
            cache_values_layers_no.append(torch.stack(values_no).to(device))


            if return_proj:
                support_feats_proj = {c: torch.stack(v, dim=0).to(device) for c, v in class_feats_proj.items()}
                support_feats_layers_proj.append(support_feats_proj)

                feat_dim_proj = feats_proj.shape[1]
                rp_proj = RobustPrototype(num_classes, feat_dim_proj, device=device)
                rp_proj.init_prototypes(support_feats_proj)
                cache_keys_layers_proj.append(rp_proj)

                values_proj = [F.one_hot(torch.tensor(class_ids.index(c)), num_classes=num_classes).float()
                               for c in class_ids]
                cache_values_layers_proj.append(torch.stack(values_proj).to(device))

    if return_proj:
        return (cache_keys_layers_no, cache_values_layers_no, support_feats_layers_no,
                cache_keys_layers_proj, cache_values_layers_proj, support_feats_layers_proj)
    else:
        return cache_keys_layers_no, cache_values_layers_no, support_feats_layers_no


# def build_support_prototypes_without_proj(model, train_loader, best_layers, device):
#     """
#     Build support prototypes for each selected layer.
    
#     Returns:
#         cache_keys_layers: list[RobustPrototype]
#         cache_values_layers: list[tensor], e.g. class one-hot
#         support_feats_layers: list[dict[class → feats]]
#     """
#     cache_keys_layers = []
#     cache_values_layers = []
#     support_feats_layers = []

#     with torch.no_grad():
#         for layer in best_layers:

#             # ------------------- Collect support features for this layer -------------------
#             class_feats = defaultdict(list)

#             for images, labels in train_loader:
#                 images, labels = images.to(device), labels.to(device)

#                 feats = extract_image_features(model, images, layers=[layer])[0]
#                 feats = F.normalize(feats, dim=-1)

#                 for i, label in enumerate(labels):
#                     class_feats[label.item()].append(feats[i])

#             # convert lists to tensors
#             class_ids = sorted(class_feats.keys())
#             support_feats = {
#                 c: torch.stack(v, dim=0).to(device)
#                 for c, v in class_feats.items()
#             }

#             support_feats_layers.append(support_feats)

#             # ------------------- Build robust prototype -------------------
#             num_classes = len(class_ids)
#             feat_dim = feats.shape[1]

#             rp = RobustPrototype(num_classes, feat_dim, device=device)
#             rp.init_prototypes(support_feats)
#             cache_keys_layers.append(rp)

#             # ------------------- Build one-hot class values -------------------
#             values = [
#                 F.one_hot(torch.tensor(class_ids.index(c)), num_classes=num_classes).float()
#                 for c in class_ids
#             ]
#             cache_values_layers.append(torch.stack(values).to(device))

#     return cache_keys_layers, cache_values_layers, support_feats_layers

# def build_support_prototypes_with_proj(model, train_loader, best_layers, device):
#     """
#     Build support prototypes for each selected layer.
    
#     Returns:
#         cache_keys_layers: list[RobustPrototype]
#         cache_values_layers: list[tensor], e.g. class one-hot
#         support_feats_layers: list[dict[class → feats]]
#     """
#     cache_keys_layers = []
#     cache_values_layers = []
#     support_feats_layers = []

#     with torch.no_grad():
#         for layer in best_layers:

#             # ------------------- Collect support features for this layer -------------------
#             class_feats = defaultdict(list)

#             for images, labels in train_loader:
#                 images, labels = images.to(device), labels.to(device)

#                 feats = extract_image_features_withproj(model, images, layers=[layer])[0]
#                 feats = F.normalize(feats, dim=-1)

#                 for i, label in enumerate(labels):
#                     class_feats[label.item()].append(feats[i])

#             # convert lists to tensors
#             class_ids = sorted(class_feats.keys())
#             support_feats = {
#                 c: torch.stack(v, dim=0).to(device)
#                 for c, v in class_feats.items()
#             }

#             support_feats_layers.append(support_feats)

#             # ------------------- Build robust prototype -------------------
#             num_classes = len(class_ids)
#             feat_dim = feats.shape[1]

#             rp = RobustPrototype(num_classes, feat_dim, device=device)
#             rp.init_prototypes(support_feats)
#             cache_keys_layers.append(rp)

#             # ------------------- Build one-hot class values -------------------
#             values = [
#                 F.one_hot(torch.tensor(class_ids.index(c)), num_classes=num_classes).float()
#                 for c in class_ids
#             ]
#             cache_values_layers.append(torch.stack(values).to(device))

#     return cache_keys_layers, cache_values_layers, support_feats_layers
    

def build_text_prototypes(text_features, cache_keys_layers, beta=20.0, weights=None):
    """ Build multi-layer adapted text features.
    text_features: [num_classes, num_prompts, feat_dim]
    cache_keys_layers: list of RobustPrototype objects, one per layer
    beta: softmax temperature for weighting prompts
    weights: list of floats for layer fusion, length = num_layers
    """
    num_classes, num_prompts, feat_dim = text_features.shape
    num_layers = len(cache_keys_layers)

    if weights is None:
        weights = [1.0 / num_layers] * num_layers

    adapted_text_feats = torch.zeros(num_classes, feat_dim, device=text_features.device)

    for c in range(num_classes):
        feats_c = F.normalize(text_features[c], dim=-1)   # [num_prompts, feat_dim]

        # For each layer, compute similarity between prototypes and text prompts
        layer_weighted = []
        for l, rp in enumerate(cache_keys_layers):
            proto_c = rp.get_prototypes()[c].unsqueeze(0)   # [1, feat_dim]
            sim = (feats_c @ proto_c.T).squeeze(-1)         # [num_prompts]
            w = torch.softmax(beta * sim, dim=0)            # prompt weights
            proto_text = (w.unsqueeze(1) * feats_c).sum(dim=0)
            layer_weighted.append(weights[l] * proto_text)

        # Fuse across layers
        fused_proto = torch.stack(layer_weighted, dim=0).sum(dim=0)
        adapted_text_feats[c] = F.normalize(fused_proto, dim=-1)

    return adapted_text_feats

def compute_robust_support_alpha(
    support_feats_layers, 
    smooth = False,
    alpha_min=0.1, 
    alpha_max=0.5, 
    eps=1e-6 ):
    """ Compute the robust support alpha.
    Keyword arguments:
    argument -- support_feats_layers: the support features for each layer
    argument -- alpha_min: the minimum alpha
    argument -- alpha_max: the maximum alpha
    argument -- eps: the epsilon value
    Return: alpha: the robust support alpha
    """
    shot_counts = []
    reliabilities = []

    for layer_dict in support_feats_layers:
        for feats in layer_dict.values():
            shot = feats.shape[0]
            shot_counts.append(shot)

            if shot < 2:
                reliabilities.append(torch.tensor(0.0, device=feats.device))
                continue

            proto = feats.mean(dim=0, keepdim=True)
            var = (feats - proto).pow(2).sum(dim=1).mean()
            reliabilities.append(torch.exp(-var))

    avg_shot = np.mean(shot_counts) if shot_counts else 1.0
    avg_reliability = float(torch.stack(reliabilities).mean()) if reliabilities else 0.0

    if smooth:
        shot_decay = 1 / np.sqrt(avg_shot + eps)   
    else:
        shot_decay = 1 / avg_shot + eps   
    shot_decay = float(np.clip(shot_decay, 0.0, 1.0))

    reliability_score = avg_reliability * (1 - shot_decay)

    alpha = alpha_min + (alpha_max - alpha_min) * (1 - reliability_score)
    alpha = float(np.clip(alpha, alpha_min, alpha_max))

    return alpha


def compute_equivalent_logit_scale(model, support_loader, text_features, device, orig_logit_scale):
    """
    Collect support zero-shot logits and compute equivalent logit_scale.

    Arguments:
    - model: loaded BiomedCLIP model
    - support_loader: DataLoader of support set
    - text_features: [num_classes, feat_dim] or [num_classes, num_prompts, feat_dim] already pooled
    - device: 'cuda' or 'cpu'
    - orig_logit_scale: original model.logit_scale.exp() value

    Returns:
    - equiv_logit_scale: scalar equivalent logit scale
    - lZS_mean: [num_classes] per-class mean of diagonal logits
    - lZS_std: [num_classes] per-class std of diagonal logits
    - support_logits: [N_support, num_classes] logits
    - support_labels: [N_support] corresponding labels
    """
    support_logits_list = []
    support_labels_list = []

    model.to(device)
    model.eval()

    # Normalize text features
    text_feats_norm = F.normalize(text_features.float().to(device), dim=-1)

    with torch.no_grad():
        for images, labels in support_loader:
            images, labels = images.to(device), labels.to(device)
            support_labels_list.append(labels.cpu())

            image_feats = model.encode_image(images).float()
            image_feats = F.normalize(image_feats, dim=-1)

            logits = model.logit_scale.exp() * (image_feats @ text_feats_norm.T)
            support_logits_list.append(logits.cpu())

    support_logits = torch.cat(support_logits_list, dim=0)
    support_labels = torch.cat(support_labels_list, dim=0).to(device)
    support_logits = support_logits.to(device)

    num_classes = support_logits.size(1)
    lZS_mean = torch.zeros(num_classes, device=device)
    lZS_std = torch.zeros(num_classes, device=device)

    for c in range(num_classes):
        mask = (support_labels == c)
        if mask.sum() == 0:
            lZS_mean[c] = 0.0
            lZS_std[c] = 1.0
        else:
            class_logits = support_logits[mask, c]  # diagonal logits
            lZS_mean[c] = class_logits.mean()
            lZS_std[c] = class_logits.std().clamp(min=1e-6)

    mean_sigma = lZS_std.mean().item()
    equiv_logit_scale = orig_logit_scale / mean_sigma

    return equiv_logit_scale



def infer_fewshot(model, test_loader, cache_keys_layers, cache_values_layers, text_feats_global, weights , equiv_logit_scale, a, beta_ot=10, device="cuda", layers=[5,7,9,11]):
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
