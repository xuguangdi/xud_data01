import os
import json
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import open_clip
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Configuration
CHECKPOINT_DIR = "checkpoints/clip_weights"
DATASET_ROOT = "data5/images"
JSON_PATH = "data5/classes.json"
RESULTS_DIR = "final_paper_ablation_exact_reprod5_multi"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-16"
HF_REPO_ID = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"

BATCH_SIZE = 32
NUM_EPOCHS = 80
NUM_REPEATS = 5
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
MAX_GRAD_NORM = 1.0


def ensure_model_weights():
    local_path = os.path.join(CHECKPOINT_DIR, "open_clip_pytorch_model.bin")
    if not os.path.exists(local_path):
        hf_hub_download(repo_id=HF_REPO_ID, filename="open_clip_pytorch_model.bin",
                        local_dir=CHECKPOINT_DIR, local_dir_use_symlinks=False, endpoint="https://hf-mirror.com")
    return local_path


def save_confusion_matrix(y_true, y_pred, classes, filename):
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = cm.astype('float') / (row_sums + 1e-12)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (row-normalized)')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


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
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class AblationModel(nn.Module):
    def __init__(self, num_classes, clip_model, feat_dim=512, mode='visual_base'):
        super().__init__()
        self.clip = clip_model
        self.feat_dim = feat_dim
        self.mode = mode
        self.num_classes = num_classes

        for param in self.clip.parameters():
            param.requires_grad = False

        self.input_norm = nn.LayerNorm(feat_dim)
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, feat_dim)
        )

        if mode != 'visual_base':
            nn.init.zeros_(self.adapter[3].weight)
            nn.init.zeros_(self.adapter[3].bias)

        self.alpha = nn.Parameter(torch.tensor(0.5))

        if self.mode == 'visual_base':
            self.head = nn.Linear(feat_dim, num_classes)
        else:
            self.register_buffer("text_prototypes", torch.zeros(num_classes, feat_dim))
            self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)
            if self.mode == 'dpsl':
                self.visual_prototypes = nn.Parameter(torch.zeros(num_classes, feat_dim))
                self.fusion_gate = nn.Parameter(torch.tensor(0.0))

    def init_components(self, tokenizer, class_names, json_path, device, train_loader=None):
        if self.mode != 'visual_base':
            kb_data = {}
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    kb_data = json.load(f)

            protos = []
            with torch.no_grad():
                for cls in class_names:
                    clean_cls = cls.replace("_", " ")
                    prompts = []
                    if self.mode == 'simple_text':
                        prompts.append(f"a photo of {clean_cls}.")
                    else:
                        prompts.append(f"a photo of {clean_cls}.")
                        prompts.append(f"a close-up photo of {clean_cls}.")
                        descs = kb_data.get(clean_cls) or kb_data.get(cls)
                        if descs:
                            if isinstance(descs, list):
                                prompts.extend(descs[:5])
                            elif isinstance(descs, str):
                                prompts.append(descs[:100])

                    toks = tokenizer(prompts).to(device)
                    f = self.clip.encode_text(toks)
                    f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)
                    protos.append((f.mean(0) / (f.mean(0).norm() + 1e-12)))
            self.text_prototypes.data = torch.stack(protos).float()

        if self.mode == 'dpsl' and train_loader is not None:
            feat_sum = torch.zeros(self.num_classes, self.feat_dim).to(device)
            counts = torch.zeros(self.num_classes).to(device)
            self.eval()
            with torch.no_grad():
                for images, labels in train_loader:
                    images = images.to(device)
                    raw = self.clip.encode_image(images)
                    raw = raw / (raw.norm(dim=-1, keepdim=True).float() + 1e-12)
                    x_norm = self.input_norm(raw)
                    x_adapt = self.adapter(x_norm)
                    v = raw + self.alpha * x_adapt
                    v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
                    for i in range(len(labels)):
                        c = labels[i].item()
                        feat_sum[c] += v[i]
                        counts[c] += 1
            self.train()
            for c in range(self.num_classes):
                if counts[c] > 0:
                    proto = feat_sum[c] / counts[c]
                    self.visual_prototypes.data[c] = proto / (proto.norm() + 1e-12)

    def forward(self, images):
        with torch.no_grad():
            raw = self.clip.encode_image(images)
            raw = raw / (raw.norm(dim=-1, keepdim=True).float() + 1e-12)
        x_norm = self.input_norm(raw)

        if self.mode == 'visual_base':
            return self.head(x_norm)
        else:
            x_adapt = self.adapter(x_norm)
            v = raw + self.alpha * x_adapt
            v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)

            if self.mode in ['simple_text', 'expert_text']:
                return self.logit_scale.exp() * torch.matmul(v, self.text_prototypes.t())
            elif self.mode == 'dpsl':
                logits_txt = torch.matmul(v, self.text_prototypes.t())
                logits_vis = torch.matmul(v, self.visual_prototypes.t())
                w = torch.sigmoid(self.fusion_gate)
                return self.logit_scale.exp() * (w * logits_txt + (1 - w) * logits_vis)


def run_one_experiment(mode, device, tokenizer, run_idx):
    weight_path = ensure_model_weights()
    model_clip, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=weight_path, device=device
    )

    train_transform = transforms.Compose([
        transforms.Resize(224), transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.TrivialAugmentWide(),
        transforms.ToTensor(), transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))
    ])

    train_ds = StrawberryDataset(DATASET_ROOT, 'train', transform=train_transform)
    init_ds = StrawberryDataset(DATASET_ROOT, 'train', transform=preprocess)
    val_ds = StrawberryDataset(DATASET_ROOT, 'val', transform=preprocess)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    init_loader = DataLoader(init_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = AblationModel(len(train_ds.classes), model_clip, mode=mode).to(device)
    model.init_components(tokenizer, train_ds.classes, JSON_PATH, device, train_loader=init_loader)

    params = []
    if mode == 'visual_base':
        params.append({'params': model.head.parameters(), 'lr': LEARNING_RATE})
    else:
        params.append({'params': model.adapter.parameters(), 'lr': LEARNING_RATE})
        params.append({'params': [model.alpha, model.logit_scale], 'lr': LEARNING_RATE})
        if mode == 'dpsl':
            params.append({'params': [model.fusion_gate], 'lr': LEARNING_RATE})
            params.append({'params': [model.visual_prototypes], 'lr': LEARNING_RATE * 0.1})

    optimizer = optim.AdamW(params, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_metrics = {'acc': 0.0, 'f1': 0.0, 'prec': 0.0, 'rec': 0.0}

    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"[{mode}] Run {run_idx} Ep {epoch + 1}/{NUM_EPOCHS}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                l1 = model(images)
                l2 = model(torch.flip(images, dims=[3]))
                logits = (l1 + l2) / 2.0
                preds = logits.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds) * 100.0
        f1 = f1_score(all_labels, all_preds, average='macro') * 100.0
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100.0
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100.0

        if acc > best_metrics['acc']:
            best_metrics = {'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec}
            cm_file = os.path.join(RESULTS_DIR, f"cm_{mode}_run{run_idx}.png")
            save_confusion_matrix(all_labels, all_preds, train_ds.classes, cm_file)

    return best_metrics


def summarize_results(per_method_runs):
    summary = {}
    for method, runs in per_method_runs.items():
        arr_acc = np.array([r['acc'] for r in runs])
        arr_f1 = np.array([r['f1'] for r in runs])
        arr_prec = np.array([r['prec'] for r in runs])
        arr_rec = np.array([r['rec'] for r in runs])
        n = len(arr_acc)

        def stats(arr):
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if n > 1 else 0.0
            median = float(np.median(arr))
            q1 = float(np.percentile(arr, 25))
            q3 = float(np.percentile(arr, 75))
            iqr = q3 - q1
            try:
                from scipy import stats as st
                ci_low, ci_high = st.t.interval(0.95, df=n - 1, loc=mean, scale=st.sem(arr))
            except Exception:
                se = std / math.sqrt(max(1, n))
                ci_low = mean - 1.96 * se
                ci_high = mean + 1.96 * se
            return {'mean': mean, 'std': std, 'median': median, 'q1': q1, 'q3': q3, 'iqr': iqr, 'ci_low': ci_low,
                    'ci_high': ci_high}

        summary[method] = {
            'acc': stats(arr_acc),
            'f1': stats(arr_f1),
            'prec': stats(arr_prec),
            'rec': stats(arr_rec),
            'n_runs': n
        }
    return summary


def paired_tests(per_method_runs, baseline_name='visual_base'):
    results = {}
    try:
        import scipy.stats as st
        have_scipy = True
    except Exception:
        have_scipy = False

    base = per_method_runs[baseline_name]
    base_acc = np.array([r['acc'] for r in base])

    for method, runs in per_method_runs.items():
        if method == baseline_name:
            continue

        arr = np.array([r['acc'] for r in runs])
        if len(arr) != len(base_acc):
            results[method] = {'paired_t': None, 'pval': None, 'cohens_d': None}
            continue

        diff = arr - base_acc
        mean_diff = diff.mean()
        sd_diff = diff.std(ddof=1)
        cohen_d = mean_diff / (sd_diff + 1e-12)

        if have_scipy:
            t_stat, pval = st.ttest_rel(arr, base_acc)
            results[method] = {'paired_t': float(t_stat), 'pval': float(pval), 'cohens_d': float(cohen_d)}
        else:
            t_stat = mean_diff / (sd_diff / math.sqrt(len(diff)) + 1e-12)
            results[method] = {'paired_t': float(t_stat), 'pval': None, 'cohens_d': float(cohen_d)}

    return results


def main():
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    methods = ['visual_base', 'simple_text', 'expert_text', 'dpsl']
    per_method_runs = {m: [] for m in methods}

    for run_idx in range(1, NUM_REPEATS + 1):
        print(f"\n=== STARTING FULL RUN {run_idx}/{NUM_REPEATS} ===")

        for method in methods:
            print(f"--- Run {run_idx} : Method {method} ---")
            metrics = run_one_experiment(method, DEVICE, tokenizer, run_idx)
            per_method_runs[method].append(metrics)

            per_run_csv = os.path.join(RESULTS_DIR, "per_run_results.csv")
            header = ["run", "method", "acc", "f1", "prec", "rec"]
            write_header = not os.path.exists(per_run_csv)

            with open(per_run_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerow([run_idx, method, metrics['acc'], metrics['f1'], metrics['prec'], metrics['rec']])

    summary = summarize_results(per_method_runs)
    paired = paired_tests(per_method_runs, baseline_name='visual_base')

    summary_csv = os.path.join(RESULTS_DIR, "summary_results.csv")
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["method", "n_runs", "acc_mean", "acc_std", "acc_median", "acc_q1", "acc_q3", "acc_ci_low", "acc_ci_high",
             "f1_mean", "f1_std", "prec_mean", "prec_std", "rec_mean", "rec_std"])
        for m, vals in summary.items():
            a = vals['acc']
            b = vals['f1']
            p = vals['prec']
            r = vals['rec']
            writer.writerow(
                [m, vals['n_runs'], a['mean'], a['std'], a['median'], a['q1'], a['q3'], a['ci_low'], a['ci_high'],
                 b['mean'], b['std'], p['mean'], p['std'], r['mean'], r['std']])

    paired_csv = os.path.join(RESULTS_DIR, "paired_tests.csv")
    with open(paired_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["vs_baseline", "t_stat", "p_value", "cohens_d"])
        for m, res in paired.items():
            writer.writerow([m, res['paired_t'], res['pval'], res['cohens_d']])

    print("\n=== AGGREGATED SUMMARY ===")
    for m, vals in summary.items():
        a = vals['acc']
        print(
            f"{m}: acc {a['mean']:.3f} ± {a['std']:.3f} (95% CI [{a['ci_low']:.3f}, {a['ci_high']:.3f}]), median {a['median']:.3f}, IQR {a['iqr']:.3f}")

    print("\n=== PAIRED TESTS (vs visual_base) ===")
    for m, res in paired.items():
        print(f"{m}: t={res['paired_t']}, p={res['pval']}, cohens_d={res['cohens_d']}")


if __name__ == "__main__":
    main()