import os
import json
import csv
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import open_clip
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from huggingface_hub import hf_hub_download
import traceback

# Configuration
warnings.filterwarnings("ignore")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
plt.switch_backend('agg')

CHECKPOINT_DIR = "checkpoints/clip_weights"
DATASET_ROOT = "data_flowers/data_flowers10"
JSON_PATH = "data_flowers/flowers.json"
RESULTS_DIR = "output_dir"
CSV_PATH = os.path.join(RESULTS_DIR, "final_comparison_metrics.csv")
PER_RUN_FILE = os.path.join(RESULTS_DIR, "per_run.csv")
DETAILED_CSV_PATH = os.path.join(RESULTS_DIR, "detailed_epoch_metrics.csv")

if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-16"
HF_REPO_ID = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"

BATCH_SIZE = 32
NUM_EPOCHS = 80
NUM_CYCLES = 5
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
MAX_GRAD_NORM = 1.0
MAX_PROMPTS_PER_CLASS = 10
EPS = 1e-12
USE_FLIP_ENSEMBLE = True


# Utilities
def ensure_model_weights():
    local_path = os.path.join(CHECKPOINT_DIR, "open_clip_pytorch_model.bin")
    if not os.path.exists(local_path):
        print("Downloading CLIP weights...")
        try:
            hf_hub_download(repo_id=HF_REPO_ID, filename="open_clip_pytorch_model.bin",
                            local_dir=CHECKPOINT_DIR, local_dir_use_symlinks=False, endpoint="https://hf-mirror.com")
        except Exception as e:
            print(f"Download failed: {e}")
    return local_path


def save_confusion_matrix(y_true, y_pred, classes, filename):
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = cm.astype('float') / (row_sums + EPS)
    plt.figure(figsize=(12, 10))
    if len(classes) > 50:
        sns.heatmap(cm_norm, annot=False, cmap='Purples')
    else:
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Purples', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_training_curves(history_dict, zs_mean, zs_std, save_path):
    plt.figure(figsize=(10, 7))
    epochs = np.arange(1, NUM_EPOCHS + 1)
    plt.axhline(y=zs_mean, color='gray', linestyle='--', label='Std Zero-Shot', alpha=0.8)

    colors = {'CoOp': '#1f77b4', 'Tip-Adapter-F': '#ff7f0e', 'CLIP-Adapter': '#2ca02c', 'DPAFNet': '#d62728'}

    for name, data_list in history_dict.items():
        if "Zero-Shot" in name: continue
        data_array = np.array(data_list)
        if data_array.size == 0 or data_array.shape[1] != NUM_EPOCHS: continue

        mean_curve = np.mean(data_array, axis=0)
        std_curve = np.std(data_array, axis=0)
        line_color = colors.get(name, 'black')

        plt.plot(epochs, mean_curve, label=name, color=line_color, linewidth=2)
        plt.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, color=line_color, alpha=0.15)

    plt.title('Flowers-102 Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_detailed_metrics(curves_dict, save_path):
    all_rows = []
    for model_name, runs_data in curves_dict.items():
        data_array = np.array(runs_data)
        if data_array.size == 0: continue
        for epoch_idx in range(NUM_EPOCHS):
            epoch_values = data_array[:, epoch_idx]
            row = {
                "Model": model_name, "Epoch": epoch_idx + 1,
                "Mean": round(np.mean(epoch_values), 4), "Std_Dev": round(np.std(epoch_values), 4)
            }
            for run_idx, val in enumerate(epoch_values): row[f"Run_{run_idx + 1}"] = round(val, 4)
            all_rows.append(row)
    if all_rows:
        pd.DataFrame(all_rows).to_csv(save_path, index=False)
        print("Detailed metrics saved.")


# Dataset
class FlowersDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths, self.labels = [], []

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Path Error: {self.root_dir}")

        self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for img in os.listdir(cls_dir):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(cls_dir, img))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
        if self.transform: image = self.transform(image)
        return image, self.labels[idx]


# Models
class StandardZeroShot(nn.Module):
    def __init__(self, num_classes, clip_model):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters(): p.requires_grad = False
        self.register_buffer("text_prototypes", torch.zeros(num_classes, 512))

    def init_prototypes(self, tokenizer, class_names, device):
        protos = []
        with torch.no_grad():
            for cls in class_names:
                clean_cls = cls.replace("_", " ")
                toks = tokenizer([f"a photo of a {clean_cls}, a type of flower."]).to(device)
                f = self.clip.encode_text(toks)
                f = f / (f.norm(dim=-1, keepdim=True) + EPS)
                protos.append(f.mean(0))
        self.text_prototypes.data = torch.stack(protos).float().to(device)

    def forward(self, images):
        with torch.no_grad():
            image_features = self.clip.encode_image(images)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + EPS)
            logits = 100.0 * image_features @ self.text_prototypes.t()
        return logits


class CoOp(nn.Module):
    def __init__(self, num_classes, clip_model, ctx_len=16):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters(): p.requires_grad = False
        self.n_ctx = ctx_len
        self.ctx_dim = clip_model.token_embedding.weight.shape[1]
        self.ctx = nn.Parameter(torch.randn(self.n_ctx, self.ctx_dim) * 0.02)
        self.register_buffer("tokenized_prompts", None)

    def init_prototypes(self, tokenizer, class_names, device):
        dummy_ctx = " ".join(["a"] * self.n_ctx)
        prompts = [f"{dummy_ctx} {cls.replace('_', ' ')}, a type of flower." for cls in class_names]
        self.tokenized_prompts = tokenizer(prompts).to(device)

    def forward(self, images):
        model_dtype = self.clip.token_embedding.weight.dtype
        with torch.no_grad():
            image_features = self.clip.encode_image(images)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + EPS)

        token_embedding = self.clip.token_embedding(self.tokenized_prompts).type(model_dtype)
        ctx_expanded = self.ctx.unsqueeze(0).expand(len(self.tokenized_prompts), -1, -1).type(model_dtype)
        prefix = token_embedding[:, :1, :]
        suffix = token_embedding[:, 1 + self.n_ctx:, :]
        prompts_embed = torch.cat([prefix, ctx_expanded, suffix], dim=1)
        prompts_embed = prompts_embed + self.clip.positional_embedding.type(model_dtype)

        x = self.clip.transformer(prompts_embed, attn_mask=self.clip.attn_mask)
        x = self.clip.ln_final(x)
        eot_indices = self.tokenized_prompts.argmax(dim=-1)
        text_features = x[torch.arange(x.shape[0]), eot_indices] @ self.clip.text_projection.type(model_dtype)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + EPS)

        logits = 100.0 * image_features @ text_features.t()
        return logits


class TipAdapterF(nn.Module):
    def __init__(self, num_classes, clip_model, feat_dim=512):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters(): p.requires_grad = False
        self.adapter = nn.Linear(feat_dim, feat_dim, bias=False)
        self.adapter.weight.data.fill_(0.0)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("cache_keys", torch.zeros(num_classes, feat_dim))
        self.register_buffer("text_prototypes", torch.zeros(num_classes, feat_dim))

    def init_cache(self, train_loader, device):
        feat_sum = torch.zeros_like(self.cache_keys).to(device)
        counts = torch.zeros(self.cache_keys.size(0)).to(device)
        with torch.no_grad():
            for imgs, lbls in train_loader:
                imgs = imgs.to(device)
                f = self.clip.encode_image(imgs)
                f = f / (f.norm(dim=-1, keepdim=True) + EPS)
                for i in range(len(lbls)):
                    feat_sum[lbls[i]] += f[i]
                    counts[lbls[i]] += 1
        for c in range(self.cache_keys.size(0)):
            if counts[c] > 0:
                proto = feat_sum[c] / counts[c]
                self.cache_keys[c].copy_(proto / (proto.norm() + EPS))

    def init_prototypes(self, tokenizer, class_names, device):
        protos = []
        with torch.no_grad():
            for cls in class_names:
                clean_cls = cls.replace("_", " ")
                prompts = [f"a photo of a {clean_cls}, a type of flower."]
                toks = tokenizer(prompts).to(device)
                f = self.clip.encode_text(toks)
                f = f / (f.norm(dim=-1, keepdim=True) + EPS)
                protos.append(f.mean(0))
        self.text_prototypes.data.copy_(torch.stack(protos).float().to(device))

    def forward(self, images):
        with torch.no_grad():
            img_feats = self.clip.encode_image(images)
            img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + EPS)
        logits_clip = 100.0 * (img_feats @ self.text_prototypes.t())
        adapter_out = self.adapter(img_feats)
        feat_fused = img_feats + adapter_out
        feat_fused = feat_fused / (feat_fused.norm(dim=-1, keepdim=True) + EPS)
        cache_logits = 100.0 * (feat_fused @ self.cache_keys.t())
        return logits_clip + self.alpha * cache_logits


class CLIPAdapter(nn.Module):
    def __init__(self, num_classes, clip_model, feat_dim=512):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters(): p.requires_grad = False
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim, bias=False)
        )
        nn.init.kaiming_normal_(self.adapter[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.adapter[2].weight)
        self.ratio = nn.Parameter(torch.tensor(0.2))
        self.register_buffer("text_prototypes", torch.zeros(num_classes, feat_dim))

    def init_prototypes(self, tokenizer, class_names, device):
        protos = []
        with torch.no_grad():
            for cls in class_names:
                clean_cls = cls.replace("_", " ")
                prompts = [f"a photo of a {clean_cls}, a type of flower."]
                toks = tokenizer(prompts).to(device)
                f = self.clip.encode_text(toks)
                f = f / (f.norm(dim=-1, keepdim=True) + EPS)
                protos.append(f.mean(0))
        self.text_prototypes.data.copy_(torch.stack(protos).float().to(device))

    def forward(self, images):
        with torch.no_grad():
            image_features = self.clip.encode_image(images)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + EPS)
        x = image_features.float()
        x_adapt = self.adapter(x)
        image_features_adapted = x + self.ratio * x_adapt
        image_features_adapted = image_features_adapted / (image_features_adapted.norm(dim=-1, keepdim=True) + EPS)
        logits = 100.0 * image_features_adapted @ self.text_prototypes.float().t()
        return logits


class DPAFNet(nn.Module):
    def __init__(self, num_classes, clip_model, feat_dim=512):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters(): p.requires_grad = False
        self.input_norm = nn.LayerNorm(feat_dim)
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, feat_dim)
        )
        nn.init.zeros_(self.adapter[3].weight)
        nn.init.zeros_(self.adapter[3].bias)

        self.fusion_gate = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("text_prototypes", torch.zeros(num_classes, feat_dim))
        self.visual_prototypes = nn.Parameter(torch.zeros(num_classes, feat_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)

    def load_knowledge(self, tokenizer, class_names, json_path, device):
        kb_data = {}
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f: kb_data = json.load(f)
        protos = []
        with torch.no_grad():
            for cls in class_names:
                clean_cls = cls.replace("_", " ")
                prompts = [f"a photo of a {clean_cls}, a type of flower."]

                descs = kb_data.get(clean_cls) or kb_data.get(cls)
                if descs and isinstance(descs, list):
                    prompts.extend(descs[:MAX_PROMPTS_PER_CLASS])

                toks = tokenizer(prompts).to(device)
                f = self.clip.encode_text(toks)
                f = f / (f.norm(dim=-1, keepdim=True) + EPS)
                protos.append(f.mean(0) / f.mean(0).norm())
        self.text_prototypes.data = torch.stack(protos).float().to(device)

    def init_visual(self, train_loader, device):
        feat_sum = torch.zeros_like(self.visual_prototypes).to(device)
        counts = torch.zeros(self.visual_prototypes.size(0)).to(device)
        self.eval()
        with torch.no_grad():
            for imgs, lbls in train_loader:
                imgs = imgs.to(device)
                raw = self.clip.encode_image(imgs)
                raw = raw / (raw.norm(dim=-1, keepdim=True).float() + EPS)
                x_norm = self.input_norm(raw)
                x_adapt = self.adapter(x_norm)
                v = raw + self.alpha * x_adapt
                v = v / (v.norm(dim=-1, keepdim=True) + EPS)
                for i in range(len(lbls)):
                    feat_sum[lbls[i]] += v[i]
                    counts[lbls[i]] += 1
        for c in range(counts.size(0)):
            if counts[c] > 0:
                proto = feat_sum[c] / counts[c]
                self.visual_prototypes.data[c].copy_(proto / (proto.norm() + EPS))
        self.train()

    def forward(self, images):
        with torch.no_grad():
            raw = self.clip.encode_image(images)
            raw = raw / (raw.norm(dim=-1, keepdim=True).float() + EPS)
        x_norm = self.input_norm(raw)
        adapter_feat = self.adapter(x_norm)
        v_final = raw + self.alpha * adapter_feat
        v_final = v_final / (v_final.norm(dim=-1, keepdim=True) + EPS)

        logits_txt = v_final @ self.text_prototypes.t()
        logits_vis = v_final @ self.visual_prototypes.t()

        w = torch.sigmoid(self.fusion_gate)
        return self.logit_scale.exp() * (w * logits_txt + (1 - w) * logits_vis)


# Training and Evaluation
def run_model(model_name, device, tokenizer, cycle_idx):
    print(f"\n{'=' * 40}\nCycle {cycle_idx} - Running: {model_name}\n{'=' * 40}")
    weight_path = ensure_model_weights()
    logging.getLogger().setLevel(logging.ERROR)
    model_clip, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=None, device=device)
    logging.getLogger().setLevel(logging.WARNING)

    try:
        checkpoint = torch.load(weight_path, map_location=device)
        if 'state_dict' in checkpoint: checkpoint = checkpoint['state_dict']
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}
        model_clip.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print(f"[FAIL] Weights: {e}")
        return 0.0, 0.0, []

    train_aug = transforms.Compose([
        transforms.Resize(224), transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.TrivialAugmentWide(),
        transforms.ToTensor(), transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))
    ])

    train_ds = FlowersDataset(DATASET_ROOT, 'train', transform=train_aug)
    init_ds = FlowersDataset(DATASET_ROOT, 'train', transform=preprocess)
    val_ds = FlowersDataset(DATASET_ROOT, 'val', transform=preprocess)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    init_loader = DataLoader(init_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    epochs = NUM_EPOCHS
    if model_name == "Zero-Shot":
        model = StandardZeroShot(len(train_ds.classes), model_clip).to(device)
        model.init_prototypes(tokenizer, train_ds.classes, device)
        epochs = 0
    elif model_name == "CoOp":
        model = CoOp(len(train_ds.classes), model_clip, ctx_len=16).to(device)
        model.init_prototypes(tokenizer, train_ds.classes, device)
    elif model_name == "Tip-Adapter-F":
        model = TipAdapterF(len(train_ds.classes), model_clip).to(device)
        model.init_prototypes(tokenizer, train_ds.classes, device)
        model.init_cache(init_loader, device)
    elif model_name == "CLIP-Adapter":
        model = CLIPAdapter(len(train_ds.classes), model_clip).to(device)
        model.init_prototypes(tokenizer, train_ds.classes, device)
    elif model_name == "DPAFNet":
        model = DPAFNet(len(train_ds.classes), model_clip).to(device)
        model.load_knowledge(tokenizer, train_ds.classes, JSON_PATH, device)
        model.init_visual(init_loader, device)
    else:
        raise ValueError("Unknown model")

    best_acc, best_f1 = 0.0, 0.0
    acc_history = []

    if epochs > 0:
        params = []
        if model_name == "DPAFNet":
            params = [
                {'params': model.adapter.parameters(), 'lr': LEARNING_RATE},
                {'params': [model.alpha, model.fusion_gate, model.logit_scale], 'lr': LEARNING_RATE},
                {'params': [model.visual_prototypes], 'lr': LEARNING_RATE * 0.1}
            ]
        elif model_name == "CoOp":
            params = [{'params': [model.ctx], 'lr': LEARNING_RATE}]
        elif model_name == "CLIP-Adapter":
            params = [
                {'params': model.adapter.parameters(), 'lr': LEARNING_RATE},
                {'params': [model.ratio], 'lr': LEARNING_RATE}
            ]
        else:
            params = filter(lambda p: p.requires_grad, model.parameters())

        optimizer = optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        for ep in range(epochs):
            model.train()
            model.clip.eval()

            loop = tqdm(train_loader, desc=f"Ep {ep + 1}/{epochs}", leave=False)
            for imgs, lbls in loop:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, lbls)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    l1 = model(imgs)
                    if USE_FLIP_ENSEMBLE:
                        l2 = model(torch.flip(imgs, dims=[3]))
                        p = ((l1 + l2) / 2.0).argmax(1)
                    else:
                        p = l1.argmax(1)
                    all_preds.extend(p.cpu().numpy())
                    all_labels.extend(lbls.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds) * 100.0
            f1 = f1_score(all_labels, all_preds, average='macro') * 100.0
            acc_history.append(acc)

            if acc > best_acc:
                best_acc, best_f1 = acc, f1
                if cycle_idx == NUM_CYCLES and model_name == "DPAFNet":
                    cm_path = os.path.join(RESULTS_DIR, f"cm_DPAFNet_Flowers.png")
                    save_confusion_matrix(all_labels, all_preds, train_ds.classes, cm_path)
    else:
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                l1 = model(imgs)
                if USE_FLIP_ENSEMBLE:
                    l2 = model(torch.flip(imgs, dims=[3]))
                    p = ((l1 + l2) / 2.0).argmax(1)
                else:
                    p = l1.argmax(1)
                all_preds.extend(p.cpu().numpy())
                all_labels.extend(lbls.cpu().numpy())
        best_acc = accuracy_score(all_labels, all_preds) * 100.0
        best_f1 = f1_score(all_labels, all_preds, average='macro') * 100.0
        acc_history = [best_acc] * NUM_EPOCHS

    print(f"[Result] {model_name} Peak Acc: {best_acc:.2f}%")
    return best_acc, best_f1, acc_history


# Main
def main():
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model_list = ["Zero-Shot", "CoOp", "Tip-Adapter-F", "CLIP-Adapter", "DPAFNet"]

    stats = {m: {"acc": [], "f1": []} for m in model_list}
    curves = {m: [] for m in model_list}

    with open(PER_RUN_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(["Cycle", "Model", "Acc", "F1"])

    for c in range(1, NUM_CYCLES + 1):
        print(f"\n########## CYCLE {c}/{NUM_CYCLES} ##########")
        for m in model_list:
            try:
                acc, f1, hist = run_model(m, DEVICE, tokenizer, c)
                stats[m]["acc"].append(acc)
                stats[m]["f1"].append(f1)
                curves[m].append(hist)
                with open(PER_RUN_FILE, 'a', newline='') as f:
                    csv.writer(f).writerow([c, m, f"{acc:.2f}", f"{f1:.2f}"])
            except Exception as e:
                print(f"[ERROR] Cycle {c} Model {m} failed: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'=' * 50}\nFINAL FLOWERS-102 RESULTS ({NUM_CYCLES} Runs)\n{'=' * 50}")
    zs_mean, zs_std = 0, 0
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Mean_Acc", "Std_Acc", "Mean_F1", "Std_F1"])
        for m in model_list:
            if not stats[m]["acc"]: continue
            accs = np.array(stats[m]["acc"])
            f1s = np.array(stats[m]["f1"])
            print(f"{m:15}: Acc={np.mean(accs):.2f}±{np.std(accs):.2f} | F1={np.mean(f1s):.2f}")
            writer.writerow(
                [m, f"{np.mean(accs):.2f}", f"{np.std(accs):.2f}", f"{np.mean(f1s):.2f}", f"{np.std(f1s):.2f}"])
            if m == "Zero-Shot": zs_mean = np.mean(accs)

    save_detailed_metrics(curves, DETAILED_CSV_PATH)
    plot_training_curves(curves, zs_mean, zs_std, os.path.join(RESULTS_DIR, "comparison_curves.png"))
    print(f"\nExperiment Complete. Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()