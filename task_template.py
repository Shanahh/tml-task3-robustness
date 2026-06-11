import argparse
import copy
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet18, resnet34, resnet50


NUM_CLASSES = 9
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Normalize(nn.Module):
    def __init__(self, mean=CIFAR_MEAN, std=CIFAR_STD):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std


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


class NormalizedModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.norm = Normalize()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(self.norm(x))

    def features(self, x):
        # ResNet feature extractor before final fc.
        x = self.norm(x)
        m = self.backbone
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def random_crop_flip(x, padding=4):
    # x: BCHW in [0,1]
    b, c, h, w = x.shape

    # horizontal flip
    flip = torch.rand(b, device=x.device) < 0.5
    x = x.clone()
    x[flip] = torch.flip(x[flip], dims=[3])

    # pad and random crop
    x_pad = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    out = torch.empty_like(x)
    for i in range(b):
        top = torch.randint(0, 2 * padding + 1, (1,), device=x.device).item()
        left = torch.randint(0, 2 * padding + 1, (1,), device=x.device).item()
        out[i] = x_pad[i, :, top:top + h, left:left + w]
    return out


def cutmix_or_mixup(x, y, num_classes, p=0.5, alpha=1.0):
    if torch.rand(1).item() > p:
        return x, y, None, 1.0, "none"

    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)

    if torch.rand(1).item() < 0.5:
        # MixUp
        mixed_x = lam * x + (1.0 - lam) * x[idx]
        return mixed_x, y, y[idx], lam, "mixup"

    # CutMix
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
    return mixed_x, y, y[idx], lam, "cutmix"


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

    model.train()
    return x_adv.detach()


def trades_attack(model, x, eps, alpha, steps):
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
    # Supervised contrastive loss over clean+adversarial/noisy features.
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)

    # remove self-comparisons
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    mask = mask * logits_mask

    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
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
        msd = self.model.state_dict()
        ssd = self.shadow.state_dict()
        for k in ssd.keys():
            if ssd[k].dtype.is_floating_point:
                ssd[k].mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)
            else:
                ssd[k].copy_(msd[k])


@torch.no_grad()
def accuracy(model, loader, device, noise_std=0.0):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
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

    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        x_adv = pgd_attack(model, x, y, eps, alpha, steps, random_start=True)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="train.npz")
    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--out", type=str, default="model.pt")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-split", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--method", type=str, default="trades",
                        choices=["clean", "pgd", "trades", "fast"])
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=7)
    parser.add_argument("--beta", type=float, default=6.0)

    parser.add_argument("--noise-std", type=float, default=0.03,
                        help="Gaussian noise training. This approximates randomized smoothing during training.")
    parser.add_argument("--supcon-weight", type=float, default=0.05,
                        help="Auxiliary supervised adversarial contrastive loss.")
    parser.add_argument("--mix-p", type=float, default=0.25,
                        help="Probability of MixUp/CutMix on clean warmup only.")
    parser.add_argument("--warmup-epochs", type=int, default=10)

    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.data)
    images = torch.from_numpy(data["images"]).float() / 255.0
    labels = torch.from_numpy(data["labels"]).long()

    assert images.ndim == 4 and images.shape[1:] == (3, 32, 32)
    assert labels.min().item() >= 0 and labels.max().item() < NUM_CLASSES

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

    backbone = build_model(args.arch)
    model = NormalizedModel(backbone).to(device)

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

    print(f"Device: {device}")
    print(f"Train size: {train_size}, Val size: {val_size}")
    print(f"Arch: {args.arch}, Method: {args.method}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x = random_crop_flip(x)

            # Warmup: clean training with MixUp/CutMix helps avoid early robust overfitting.
            use_clean_warmup = epoch <= args.warmup_epochs

            if use_clean_warmup:
                x_train, y_a, y_b, lam, _ = cutmix_or_mixup(
                    x, y, NUM_CLASSES, p=args.mix_p, alpha=1.0
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
                    # Fast adversarial training: random-start FGSM.
                    x_adv = pgd_attack(
                        model, x_noisy, y,
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
                        model, x_noisy, y,
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
                            f_clean = model.features(x_noisy)
                            f_adv = model.features(x_adv)
                            f_all = torch.cat([f_clean, f_adv], dim=0)
                            y_all = torch.cat([y, y], dim=0)
                            loss = loss + args.supcon_weight * supervised_contrastive_loss(f_all, y_all)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                elif args.method == "trades":
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(enabled=False):
                        # Attack generation should stay in fp32 for stable gradients.
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
                            f_clean = model.features(x_noisy)
                            f_adv = model.features(x_adv)
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

        # Proxy for the hidden score
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

            if ema is not None:
                state = copy.deepcopy(ema.shadow.backbone.state_dict())
            else:
                state = copy.deepcopy(model.backbone.state_dict())
            best_state = copy.deepcopy(state)
            torch.save(state, args.out)
            print(f"Saved best checkpoint to {args.out}")

    if best_state is None:
        print("WARNING: no checkpoint saved because clean validation accuracy never exceeded 50%.")
        print("Try more epochs, lower eps for early training, or resnet34.")
    else:
        torch.save(best_state, args.out)
        print(f"Final best proxy score: {best_score_proxy:.4f}")
        print(f"Saved state_dict only: {args.out}")

    # Submission sanity check
    test_backbone = build_model(args.arch)
    test_backbone.load_state_dict(torch.load(args.out, map_location="cpu"))
    test_backbone.eval()
    with torch.no_grad():
        out = test_backbone(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, NUM_CLASSES), out.shape
    print("Sanity check passed: output shape is", tuple(out.shape))


if __name__ == "__main__":
    main()
