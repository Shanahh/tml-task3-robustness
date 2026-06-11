import argparse
import copy
import math
import os
import random
import zipfile
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet18, resnet34, resnet50


NUM_CLASSES = 9


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    print("Dataset path:", path)
    print("Dataset size:", os.path.getsize(path), "bytes")

    if not zipfile.is_zipfile(path):
        with open(path, "rb") as f:
            prefix = f.read(300)
        raise ValueError(
            "This is not a valid .npz archive. "
            "You probably have a Git LFS pointer, HTML page, redirect file, or corrupted download.\n"
            f"First bytes:\n{prefix!r}"
        )

    data = np.load(path, allow_pickle=False)

    print("NPZ keys:", data.files)

    if "images" not in data.files or "labels" not in data.files:
        raise KeyError(f"Expected keys ['images', 'labels'], got {data.files}")

    images = data["images"]
    labels = data["labels"]

    print("images:", images.shape, images.dtype, images.min(), images.max())
    print("labels:", labels.shape, labels.dtype, labels.min(), labels.max())

    assert images.ndim == 4 and images.shape[1:] == (3, 32, 32), images.shape
    assert images.dtype == np.uint8, images.dtype
    assert labels.min() >= 0 and labels.max() < NUM_CLASSES

    return images, labels


def build_model(arch: str):
    constructors = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
    }

    if arch not in constructors:
        raise ValueError(f"Unsupported architecture: {arch}")

    model = constructors[arch](weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def resnet_features(model, x):
    # Latent representation before the final fc layer.
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


def random_crop_flip(x, padding=4):
    # x: BCHW in [0, 1]
    b, c, h, w = x.shape

    flip = torch.rand(b, device=x.device) < 0.5
    x = x.clone()
    x[flip] = torch.flip(x[flip], dims=[3])

    x_pad = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    out = torch.empty_like(x)

    for i in range(b):
        top = torch.randint(0, 2 * padding + 1, (1,), device=x.device).item()
        left = torch.randint(0, 2 * padding + 1, (1,), device=x.device).item()
        out[i] = x_pad[i, :, top:top + h, left:left + w]

    return out


def cutmix_or_mixup(x, y, p=0.5, alpha=1.0):
    if torch.rand(1).item() > p:
        return x, y, None, 1.0

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)

    if torch.rand(1).item() < 0.5:
        mixed_x = lam * x + (1.0 - lam) * x[idx]
        return mixed_x, y, y[idx], lam

    b, c, h, w = x.shape
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = torch.randint(w, (1,), device=x.device).item()
    cy = torch.randint(h, (1,), device=x.device).item()

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]

    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
    return mixed_x, y, y[idx], lam


def mixed_ce_loss(logits, y_a, y_b=None, lam=1.0):
    if y_b is None:
        return F.cross_entropy(logits, y_a, label_smoothing=0.05)

    return (
        lam * F.cross_entropy(logits, y_a, label_smoothing=0.05)
        + (1.0 - lam) * F.cross_entropy(logits, y_b, label_smoothing=0.05)
    )


@torch.no_grad()
def clamp_linf(x_adv, x_nat, eps):
    return torch.max(torch.min(x_adv, x_nat + eps), x_nat - eps).clamp(0.0, 1.0)


def pgd_attack(model, x, y, eps, alpha, steps, random_start=True):
    was_training = model.training
    model.eval()

    if random_start:
        x_adv = x.detach() + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0.0, 1.0)
    else:
        x_adv = x.detach().clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = clamp_linf(x_adv, x, eps)

    if was_training:
        model.train()

    return x_adv.detach()


def trades_attack(model, x, eps, alpha, steps):
    was_training = model.training
    model.eval()

    with torch.no_grad():
        p_nat = F.softmax(model(x), dim=1)

    x_adv = x.detach() + 0.001 * torch.randn_like(x)
    x_adv = x_adv.clamp(0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        log_p_adv = F.log_softmax(model(x_adv), dim=1)
        loss_kl = F.kl_div(log_p_adv, p_nat, reduction="batchmean")
        grad = torch.autograd.grad(loss_kl, x_adv, only_inputs=True)[0]

        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = clamp_linf(x_adv, x, eps)

    if was_training:
        model.train()

    return x_adv.detach()


def trades_loss(model, x, y, eps, alpha, steps, beta):
    x_adv = trades_attack(model, x, eps, alpha, steps)

    logits_nat = model(x)
    logits_adv = model(x_adv)

    loss_nat = F.cross_entropy(logits_nat, y, label_smoothing=0.05)
    loss_rob = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_nat.detach(), dim=1),
        reduction="batchmean",
    )

    return loss_nat + beta * loss_rob, logits_nat, x_adv


def supervised_contrastive_loss(features, labels, temperature=0.2):
    features = F.normalize(features, dim=1)

    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)

    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positives = mask.sum(dim=1)
    valid = positives > 0

    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positives + 1e-12)
    return -mean_log_prob_pos[valid].mean()


@dataclass
class EMA:
    model: nn.Module
    decay: float = 0.999

    def __post_init__(self):
        self.shadow = copy.deepcopy(self.model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self):
        model_state = self.model.state_dict()
        shadow_state = self.shadow.state_dict()

        for k in shadow_state.keys():
            if shadow_state[k].dtype.is_floating_point:
                shadow_state[k].mul_(self.decay).add_(model_state[k], alpha=1.0 - self.decay)
            else:
                shadow_state[k].copy_(model_state[k])


@torch.no_grad()
def accuracy(model, loader, device, noise_std=0.0):
    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if noise_std > 0:
            x = (x + noise_std * torch.randn_like(x)).clamp(0.0, 1.0)

        pred = model(x).argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

    return correct / max(total, 1)


def robust_accuracy_pgd(model, loader, device, eps, alpha, steps, max_batches=20):
    model.eval()

    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_adv = pgd_attack(model, x, y, eps, alpha, steps, random_start=True)

        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

    return correct / max(total, 1)


def save_checkpoint(path, state):
    torch.save(state, path)

    loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise RuntimeError("Saved file is not a state dict.")

    print(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="train.npz")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--out", type=str, default="model.pt")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-split", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--method", type=str, default="trades", choices=["clean", "pgd", "trades", "fast"])
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=7)
    parser.add_argument("--beta", type=float, default=6.0)

    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--supcon-weight", type=float, default=0.05)
    parser.add_argument("--mix-p", type=float, default=0.25)
    parser.add_argument("--warmup-epochs", type=int, default=10)

    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)

    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_np, labels_np = load_npz_dataset(args.data)

    images = torch.from_numpy(images_np).float() / 255.0
    labels = torch.from_numpy(labels_np).long()

    dataset = TensorDataset(images, labels)

    val_size = args.val_split
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = build_model(args.arch).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    scaler = GradScaler(enabled=args.amp)
    ema = EMA(model, decay=args.ema_decay) if args.ema else None

    best_score_proxy = -1.0
    best_state = None
    best_clean = 0.0
    best_pgd = 0.0

    print(f"Device: {device}")
    print(f"Train size: {train_size}")
    print(f"Val size: {val_size}")
    print(f"Arch: {args.arch}")
    print(f"Method: {args.method}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        running_acc = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x = random_crop_flip(x)

            use_clean_warmup = epoch <= args.warmup_epochs

            if use_clean_warmup:
                x_train, y_a, y_b, lam = cutmix_or_mixup(
                    x,
                    y,
                    p=args.mix_p,
                    alpha=1.0,
                )

                if args.noise_std > 0:
                    x_train = (x_train + args.noise_std * torch.randn_like(x_train)).clamp(0.0, 1.0)

                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=args.amp):
                    logits = model(x_train)
                    loss = mixed_ce_loss(logits, y_a, y_b, lam)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                if args.noise_std > 0:
                    x_noisy = (x + args.noise_std * torch.randn_like(x)).clamp(0.0, 1.0)
                else:
                    x_noisy = x

                if args.method == "clean":
                    optimizer.zero_grad(set_to_none=True)

                    with autocast(enabled=args.amp):
                        logits = model(x_noisy)
                        loss = F.cross_entropy(logits, y, label_smoothing=0.05)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                elif args.method == "fast":
                    x_adv = pgd_attack(
                        model,
                        x_noisy,
                        y,
                        eps=args.eps,
                        alpha=args.eps * 1.25,
                        steps=1,
                        random_start=True,
                    )

                    optimizer.zero_grad(set_to_none=True)

                    with autocast(enabled=args.amp):
                        logits = model(x_adv)
                        loss = F.cross_entropy(logits, y, label_smoothing=0.05)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                elif args.method == "pgd":
                    x_adv = pgd_attack(
                        model,
                        x_noisy,
                        y,
                        eps=args.eps,
                        alpha=args.alpha,
                        steps=args.pgd_steps,
                        random_start=True,
                    )

                    optimizer.zero_grad(set_to_none=True)

                    with autocast(enabled=args.amp):
                        logits_clean = model(x_noisy)
                        logits_adv = model(x_adv)

                        loss = (
                            0.25 * F.cross_entropy(logits_clean, y, label_smoothing=0.05)
                            + 0.75 * F.cross_entropy(logits_adv, y, label_smoothing=0.05)
                        )

                        if args.supcon_weight > 0:
                            f_clean = resnet_features(model, x_noisy)
                            f_adv = resnet_features(model, x_adv)
                            f_all = torch.cat([f_clean, f_adv], dim=0)
                            y_all = torch.cat([y, y], dim=0)
                            loss = loss + args.supcon_weight * supervised_contrastive_loss(f_all, y_all)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                elif args.method == "trades":
                    optimizer.zero_grad(set_to_none=True)

                    with autocast(enabled=False):
                        loss_trades, logits, x_adv = trades_loss(
                            model,
                            x_noisy,
                            y,
                            eps=args.eps,
                            alpha=args.alpha,
                            steps=args.pgd_steps,
                            beta=args.beta,
                        )

                    with autocast(enabled=args.amp):
                        loss = loss_trades

                        if args.supcon_weight > 0:
                            f_clean = resnet_features(model, x_noisy)
                            f_adv = resnet_features(model, x_adv)
                            f_all = torch.cat([f_clean, f_adv], dim=0)
                            y_all = torch.cat([y, y], dim=0)
                            loss = loss + args.supcon_weight * supervised_contrastive_loss(f_all, y_all)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            if ema is not None:
                ema.update()

            with torch.no_grad():
                pred = model(x).argmax(dim=1)
                running_acc += (pred == y).sum().item()
                running_loss += loss.item() * y.size(0)
                seen += y.size(0)

        scheduler.step()

        eval_model = ema.shadow if ema is not None else model

        clean_acc = accuracy(eval_model, val_loader, device)
        smooth_acc = accuracy(eval_model, val_loader, device, noise_std=args.noise_std)
        pgd_acc = robust_accuracy_pgd(
            eval_model,
            val_loader,
            device,
            eps=args.eps,
            alpha=args.alpha,
            steps=max(args.pgd_steps, 10),
            max_batches=20,
        )

        score_proxy = 0.5 * clean_acc + 0.5 * pgd_acc

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={running_loss / seen:.4f} | "
            f"train_acc={running_acc / seen:.4f} | "
            f"val_clean={clean_acc:.4f} | "
            f"val_smooth={smooth_acc:.4f} | "
            f"val_pgd={pgd_acc:.4f} | "
            f"proxy={score_proxy:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.5f}"
        )

        if clean_acc > 0.50 and score_proxy > best_score_proxy:
            best_score_proxy = score_proxy
            best_clean = clean_acc
            best_pgd = pgd_acc

            state = copy.deepcopy(eval_model.state_dict())
            best_state = copy.deepcopy(state)

            save_checkpoint(args.out, best_state)

    if best_state is None:
        raise RuntimeError(
            "No checkpoint saved because clean validation accuracy never exceeded 50%. "
            "Do not submit."
        )

    save_checkpoint(args.out, best_state)

    print(f"Best proxy score: {best_score_proxy:.4f}")
    print(f"Best clean validation accuracy: {best_clean:.4f}")
    print(f"Best PGD validation accuracy: {best_pgd:.4f}")

    # Submission sanity check:
    # This EXACTLY mirrors what the server is likely doing for model construction.
    test_model = build_model(args.arch).to(device)
    state = torch.load(args.out, map_location=device)

    missing, unexpected = test_model.load_state_dict(state, strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    assert len(missing) == 0, missing
    assert len(unexpected) == 0, unexpected

    test_model.eval()

    with torch.no_grad():
        out = test_model(torch.randn(1, 3, 32, 32, device=device))

    assert out.shape == (1, NUM_CLASSES), out.shape
    print("Sanity check passed: output shape is", tuple(out.shape))

    submitted_clean_acc = accuracy(test_model, val_loader, device)
    print(f"Submitted plain-model validation clean accuracy: {submitted_clean_acc:.4f}")

    if submitted_clean_acc <= 0.50:
        raise RuntimeError(
            "Saved plain model has <=50% clean validation accuracy. "
            "Do not submit this checkpoint."
        )


if __name__ == "__main__":
    main()