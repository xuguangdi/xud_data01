import os
import csv
import json
from PIL import Image
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as tvmodels

import open_clip
from huggingface_hub import hf_hub_download

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Configuration
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
plt.switch_backend('agg')

CHECKPOINT_DIR = "checkpoints/clip_weights"
DATASET_ROOT = "data10/images"
# JSON_PATH = "data6/classes.json"
RESULTS_DIR = "final_visual_vs_dpafnet5_curves"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "ViT-B-16"
HF_REPO_ID = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"

BATCH_SIZE = 32
NUM_EPOCHS = 80
NUM_CYCLES = 5
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_DP = 5e-4
WEIGHT_DECAY = 1e-3
MAX_GRAD_NORM = 1.0


def ensure_model_weights():
    local_path = os.path.join(CHECKPOINT_DIR, "open_clip_pytorch_model.bin")
    if not os.path.exists(local_path):
        print("Downloading CLIP weights to", CHECKPOINT_DIR)
        hf_hub_download(repo_id=HF_REPO_ID, filename="open_clip_pytorch_model.bin",
                        local_dir=CHECKPOINT_DIR, local_dir_use_symlinks=False, endpoint="https://hf-mirror.com")
    return local_path


def save_confusion_matrix(y_true, y_pred, classes, filename):
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = cm.astype('float') / (row_sums + 1e-12)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (normalized by row)')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_comparison_curves(history_dict, save_path):
    plt.figure(figsize=(10, 7))

    colors = {
        'DPAFNet': '#d62728',
        'ResNet50': '#1f77b4',
        'EfficientNet_B0': '#2ca02c',
        'ViT_B_16': '#ff7f0e'
    }

    first_key = list(history_dict.keys())[0]
    epochs = np.arange(1, len(history_dict[first_key][0]) + 1)

    for name, runs_data in history_dict.items():
        data_array = np.array(runs_data)
        if data_array.size == 0: continue

        mean_curve = np.mean(data_array, axis=0)
        std_curve = np.std(data_array, axis=0)
        c = colors.get(name, 'black')

        plt.plot(epochs, mean_curve, label=name, color=c, linewidth=2, linestyle='-')
        plt.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, color=c, alpha=0.15)

    plt.title(f'Validation Accuracy Dynamics ({NUM_CYCLES}-Run Avg)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Curves saved to {save_path}")


class StrawberryDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths, self.labels = [], []

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Path Error: {self.root_dir}")

        self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(cls_dir, fn))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def build_torchvision_backbone(name, device):
    """
    Return (backbone_model, feature_dim).
    Backbone forward outputs feature vector (B, feat_dim) with classifier/head removed.
    """
    name_lower = name.lower()
    if name_lower == "resnet50":
        model = tvmodels.resnet50(pretrained=True)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name_lower == "efficientnet_b0" or name_lower == "efficientnetb0":
        model = tvmodels.efficientnet_b0(pretrained=True)
        last = model.classifier
        if isinstance(last, nn.Sequential):
            feat_dim = last[-1].in_features
        else:
            feat_dim = last.in_features
        model.classifier = nn.Identity()
    elif name_lower == "vit_b_16" or name_lower == "vit_b16":
        model = tvmodels.vit_b_16(pretrained=True)
        if hasattr(model, 'heads'):
            try:
                last = model.heads
                if isinstance(last, nn.Sequential):
                    feat_dim = last[-1].in_features
                elif isinstance(last, nn.Linear):
                    feat_dim = last.in_features
                else:
                    feat_dim = getattr(model, 'hidden_dim', getattr(model, 'embed_dim', 768))
                model.heads = nn.Identity()
            except Exception:
                feat_dim = getattr(model, 'hidden_dim', getattr(model, 'embed_dim', 768))
                model.heads = nn.Identity()
        else:
            feat_dim = getattr(model, 'hidden_dim', getattr(model, 'embed_dim', 768))
    else:
        raise ValueError(f"Unknown backbone name: {name}")

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224, device=device)
        feat = model(dummy)
        if feat.ndim == 4:
            feat = feat.mean(dim=[2, 3])
        feat_dim_inferred = feat.shape[1]

    return model, feat_dim_inferred


class VisualLinearProbe(nn.Module):
    def __init__(self, backbone, feat_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, images):
        feats = self.backbone(images)
        if feats.ndim == 4:
            feats = feats.mean(dim=[2, 3])
        return self.head(feats)


class DPAFNet(nn.Module):
    def __init__(self, num_classes, clip_model, feat_dim=512):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters(): p.requires_grad = False
        self.input_norm = nn.LayerNorm(feat_dim)
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, feat_dim)
        )
        nn.init.zeros_(self.adapter[3].weight)
        nn.init.zeros_(self.adapter[3].bias)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("text_prototypes", torch.zeros(num_classes, feat_dim))
        self.visual_prototypes = nn.Parameter(torch.zeros(num_classes, feat_dim))
        self.fusion_gate = nn.Parameter(torch.tensor(0.0))
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)

    def load_knowledge(self, tokenizer, class_names, json_path, device):
        kb_data = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    kb_data = json.load(f)
            except:
                kb_data = {}
        protos = []
        with torch.no_grad():
            for cls in class_names:
                clean_cls = cls.replace("_", " ")
                prompts = [f"a photo of {clean_cls}.", f"a close-up photo of {clean_cls}, a strawberry disease."]
                if cls in kb_data: prompts.extend(kb_data[cls][:5])
                toks = tokenizer(prompts).to(device)
                f = self.clip.encode_text(toks)
                f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)
                avg = f.mean(0)
                avg = avg / (avg.norm() + 1e-12)
                protos.append(avg)
        self.text_prototypes.data.copy_(torch.stack(protos).float().to(device))

    def init_visual(self, train_loader, device):
        feat_sum = torch.zeros_like(self.visual_prototypes).to(device)
        counts = torch.zeros(self.visual_prototypes.size(0)).to(device)
        self.eval()
        with torch.no_grad():
            for imgs, lbls in tqdm(train_loader, desc="InitVisualProto", leave=False):
                imgs = imgs.to(device)
                raw = self.clip.encode_image(imgs)
                raw = raw / (raw.norm(dim=-1, keepdim=True) + 1e-12)
                x_norm = self.input_norm(raw)
                x_adapt = self.adapter(x_norm)
                v = raw + self.alpha * x_adapt
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
                for i in range(len(lbls)):
                    feat_sum[lbls[i]] += v[i]
                    counts[lbls[i]] += 1
        self.train()
        for c in range(counts.size(0)):
            if counts[c] > 0:
                proto = feat_sum[c] / counts[c]
                proto = proto / (proto.norm() + 1e-12)
                self.visual_prototypes.data[c].copy_(proto)

    def forward(self, images):
        with torch.no_grad():
            raw = self.clip.encode_image(images)
            raw = raw / (raw.norm(dim=-1, keepdim=True) + 1e-12)
        x_norm = self.input_norm(raw)
        adapter_feat = self.adapter(x_norm)
        v_final = raw + self.alpha * adapter_feat
        v_final = v_final / (v_final.norm(dim=-1, keepdim=True) + 1e-12)
        logits_txt = v_final @ self.text_prototypes.t()
        logits_vis = v_final @ self.visual_prototypes.t()
        w = torch.sigmoid(self.fusion_gate)
        return self.logit_scale.exp() * (w * logits_txt + (1 - w) * logits_vis)


def train_one_model(model_name, device, tokenizer=None, clip_model=None, preprocess_for_clip=None):
    print("\n" + "=" * 40)
    print("Running model:", model_name)
    print("=" * 40)

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    if preprocess_for_clip is None:
        val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        val_transform = preprocess_for_clip

    train_ds = StrawberryDataset(DATASET_ROOT, 'val', transform=train_transform)
    val_ds = StrawberryDataset(DATASET_ROOT, 'train', transform=val_transform)
    init_ds = StrawberryDataset(DATASET_ROOT, 'val', transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    init_loader = DataLoader(init_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(train_ds.classes)

    model = None
    saved_model_path = os.path.join(RESULTS_DIR, f"best_{model_name}.pth")
    saved_cm_path = os.path.join(RESULTS_DIR, f"cm_{model_name}.png")

    if model_name == "DPAFNet":
        if clip_model is None or tokenizer is None:
            raise RuntimeError("DPAFNet needs CLIP model and tokenizer passed in")
        model = DPAFNet(num_classes, clip_model).to(device)
        model.load_knowledge(tokenizer, train_ds.classes, JSON_PATH, device)
        model.init_visual(init_loader, device)
        opt = optim.AdamW([
            {'params': model.adapter.parameters(), 'lr': LEARNING_RATE_DP},
            {'params': [model.alpha, model.logit_scale, model.fusion_gate], 'lr': LEARNING_RATE_DP},
            {'params': [model.visual_prototypes], 'lr': LEARNING_RATE_DP * 0.1}
        ], weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    else:
        if model_name == "ResNet50":
            backbone, feat_dim = build_torchvision_backbone("resnet50", device)
        elif model_name == "EfficientNet_B0":
            backbone, feat_dim = build_torchvision_backbone("efficientnet_b0", device)
        elif model_name == "ViT_B_16":
            backbone, feat_dim = build_torchvision_backbone("vit_b_16", device)
        else:
            raise ValueError("Unknown visual baseline: " + str(model_name))

        model = VisualLinearProbe(backbone, feat_dim, num_classes).to(device)
        opt = optim.AdamW(model.head.parameters(), lr=LEARNING_RATE_HEAD, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    best_f1 = 0.0
    acc_history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"[{model_name}] Ep {epoch + 1}/{NUM_EPOCHS}", leave=False)
        for imgs, lbls in loop:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            opt.step()
            loop.set_postfix(loss=loss.item())
        scheduler.step()

        # Validation (TTA)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                l1 = model(imgs)
                l2 = model(torch.flip(imgs, dims=[3]))
                logits = (l1 + l2) / 2.0
                preds = logits.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbls.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds) * 100.0
        f1 = f1_score(all_labels, all_preds, average='macro') * 100.0

        acc_history.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            try:
                torch.save(model.state_dict(), saved_model_path)
            except Exception as e:
                print("Warning saving model:", e)
            save_confusion_matrix(all_labels, all_preds, train_ds.classes, saved_cm_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{model_name}] Ep {epoch + 1}: Val Acc {acc:.2f}% (Best {best_acc:.2f}%)")

    return best_acc, best_f1, saved_model_path, saved_cm_path, acc_history


def main():
    print("Preparing CLIP model for DPAFNet...")
    weight_path = ensure_model_weights()
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=weight_path,
                                                                           device=DEVICE)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    models_list = ["ResNet50", "EfficientNet_B0", "ViT_B_16", "DPAFNet"]
    all_curves = {m: [] for m in models_list}

    summary_path = os.path.join(RESULTS_DIR, "summary_results.csv")
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Cycle", "Model", "Final_Acc", "Final_F1", "SavedModelPath"])

    for cycle in range(1, NUM_CYCLES + 1):
        print(f"\n{'#' * 20} CYCLE {cycle}/{NUM_CYCLES} {'#' * 20}")

        for name in models_list:
            if name == "DPAFNet":
                acc, f1, mpath, cmpath, hist = train_one_model(name, DEVICE, tokenizer=tokenizer, clip_model=clip_model,
                                                               preprocess_for_clip=clip_preprocess)
            else:
                acc, f1, mpath, cmpath, hist = train_one_model(name, DEVICE)

            all_curves[name].append(hist)

            with open(summary_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([cycle, name, f"{acc:.4f}", f"{f1:.4f}", mpath])

    print("\nPlotting combined curves...")
    plot_comparison_curves(all_curves, os.path.join(RESULTS_DIR, "final_comparison_curves.png"))

    print("\nSaving detailed curve data to CSV...")
    detailed_csv_path = os.path.join(RESULTS_DIR, "detailed_training_curves.csv")
    with open(detailed_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Model", "Epoch"] + [f"Run_{i}" for i in range(1, NUM_CYCLES + 1)] + ["Mean", "Std_Dev", "Variance"]
        writer.writerow(header)

        for name, runs_data in all_curves.items():
            data_array = np.array(runs_data)
            if data_array.size == 0: continue

            means = np.mean(data_array, axis=0)
            stds = np.std(data_array, axis=0)
            vars = np.var(data_array, axis=0)

            num_epochs = data_array.shape[1]
            for ep in range(num_epochs):
                raw_values = data_array[:, ep]
                row = [name, ep + 1] + raw_values.tolist() + [means[ep], stds[ep], vars[ep]]
                writer.writerow(row)

    print(f"Detailed data saved to {detailed_csv_path}")
    print("\nAll done. Summary written to", summary_path)
    print("Confusion matrices and curves are in:", RESULTS_DIR)


if __name__ == "__main__":
    main()