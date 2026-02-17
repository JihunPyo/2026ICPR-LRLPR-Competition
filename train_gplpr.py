import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
from train_funcs.train_utils import strLabelConverter

try:
    import wandb
except ImportError:
    wandb = None


def make_dataloader(spec, tag: str):
    dataset = datasets.make(spec["dataset"])
    wrapper = datasets.make(spec["wrapper"], args={"dataset": dataset})
    return DataLoader(
        wrapper,
        batch_size=spec["batch"],
        shuffle=(tag == "train"),
        num_workers=spec.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        collate_fn=wrapper.collate_fn,
    )


def compute_batch_loss_and_preds(model, hr, gt_texts, converter, criterion, max_len: int, device):
    _, logits, _ = model(hr.to(device))
    target = converter.encode_list(gt_texts, K=max_len).to(device)
    loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

    pred_idx = logits.argmax(2).detach().cpu()
    pred_texts = converter.decode_list(pred_idx)
    pred_texts = [p.replace("-", "") for p in pred_texts]
    gt_texts = [g.replace("-", "") for g in gt_texts]
    correct = sum(int(p == g) for p, g in zip(pred_texts, gt_texts))

    total_ed = 0
    total_chars = 0
    for pred, gt in zip(pred_texts, gt_texts):
        total_ed += edit_distance(pred, gt)
        total_chars += max(1, len(gt))

    return loss, correct, len(gt_texts), total_ed, total_chars


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def save_checkpoint(save_dir: Path, epoch: int, model, optimizer, best_acc: float, config: dict, name: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {
            "name": config["MODEL_OCR"]["name"],
            "args": config["MODEL_OCR"]["args"],
            "sd": model.state_dict(),
        },
        "optimizer": {
            "name": config["optimizer_ocr"]["name"],
            "args": config["optimizer_ocr"]["args"],
            "sd": optimizer.state_dict(),
        },
        "epoch": epoch,
        "state": torch.get_rng_state(),
        "best_acc": best_acc,
    }
    torch.save(payload, save_dir / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(config["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"gplpr-{save_dir.name}"

    train_loader = make_dataloader(config["train_dataset"], "train")
    val_loader = make_dataloader(config["val_dataset"], "val")

    model = models.make(config["MODEL_OCR"]).to(device)
    converter = strLabelConverter(config["alphabet"])
    max_len = int(config["MODEL_OCR"]["args"].get("K", 7))
    criterion = nn.CrossEntropyLoss()
    opt_name = config["optimizer_ocr"]["name"].lower()
    opt_args = config["optimizer_ocr"].get("args", {})
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **opt_args)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), **opt_args)
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer_ocr']['name']}")
    scheduler = StepLR(
        optimizer,
        step_size=config["train"].get("lr_step", 10),
        gamma=config["train"].get("lr_gamma", 0.5),
    )

    metrics_csv = save_dir / "metrics.csv"
    summary_json = save_dir / "summary.json"
    if not metrics_csv.exists():
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "lr",
                    "train_loss",
                    "train_acc",
                    "train_avg_ed",
                    "train_cer",
                    "val_loss",
                    "val_acc",
                    "val_avg_ed",
                    "val_cer",
                    "best_val_acc",
                ]
            )

    best_val_acc = 0.0
    epochs = int(config["train"]["epochs"])
    wandb_cfg = config.get("wandb", {})
    use_wandb = bool(wandb_cfg.get("enabled", False))
    if use_wandb and wandb is None:
        print("[WARN] wandb is enabled in config but not installed; continue without wandb.")
        use_wandb = False
    if use_wandb:
        try:
            wandb.init(
                project=wandb_cfg.get("project", "gplpr"),
                entity=wandb_cfg.get("entity"),
                name=wandb_cfg.get("run_name", run_name),
                config=config,
                mode=wandb_cfg.get("mode", "online"),
                dir=wandb_cfg.get("dir"),
            )
        except Exception as e:
            print(f"[WARN] wandb init failed; continue without wandb: {e}")
            use_wandb = False

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        train_ed_sum = 0
        train_char_sum = 0

        pbar = tqdm(train_loader, desc=f"train {epoch}/{epochs}", leave=False)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            loss, correct, count, ed_sum, char_sum = compute_batch_loss_and_preds(
                model=model,
                hr=batch["hr"],
                gt_texts=batch["gt"],
                converter=converter,
                criterion=criterion,
                max_len=max_len,
                device=device,
            )
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * count
            train_correct += correct
            train_count += count
            train_ed_sum += ed_sum
            train_char_sum += char_sum
            pbar.set_postfix(loss=f"{(train_loss_sum / max(train_count, 1)):.4f}")

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0
        val_ed_sum = 0
        val_char_sum = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val {epoch}/{epochs}", leave=False):
                loss, correct, count, ed_sum, char_sum = compute_batch_loss_and_preds(
                    model=model,
                    hr=batch["hr"],
                    gt_texts=batch["gt"],
                    converter=converter,
                    criterion=criterion,
                    max_len=max_len,
                    device=device,
                )
                val_loss_sum += loss.item() * count
                val_correct += correct
                val_count += count
                val_ed_sum += ed_sum
                val_char_sum += char_sum

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        train_acc = train_correct / max(train_count, 1)
        val_acc = val_correct / max(val_count, 1)
        train_avg_ed = train_ed_sum / max(train_count, 1)
        val_avg_ed = val_ed_sum / max(val_count, 1)
        train_cer = train_ed_sum / max(train_char_sum, 1)
        val_cer = val_ed_sum / max(val_char_sum, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(save_dir, epoch, model, optimizer, best_val_acc, config, "best_gplpr.pth")
        save_checkpoint(save_dir, epoch, model, optimizer, best_val_acc, config, "last_gplpr.pth")

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    lr_now,
                    train_loss,
                    train_acc,
                    train_avg_ed,
                    train_cer,
                    val_loss,
                    val_acc,
                    val_avg_ed,
                    val_cer,
                    best_val_acc,
                ]
            )

        summary = {
            "last_epoch": epoch,
            "best_val_acc": best_val_acc,
            "latest": {
                "lr": lr_now,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_avg_ed": train_avg_ed,
                "train_cer": train_cer,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_avg_ed": val_avg_ed,
                "val_cer": val_cer,
            },
            "config_path": str(Path(args.config).resolve()),
            "save_dir": str(save_dir.resolve()),
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if use_wandb:
            try:
                wandb.log(
                    {
                        "epoch": epoch,
                        "lr": lr_now,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "train_avg_ed": train_avg_ed,
                        "train_cer": train_cer,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_avg_ed": val_avg_ed,
                        "val_cer": val_cer,
                        "best_val_acc": best_val_acc,
                    },
                    step=epoch,
                )
            except Exception as e:
                print(f"[WARN] wandb log failed; disable wandb: {e}")
                use_wandb = False

        print(
            f"epoch {epoch}/{epochs} | lr={lr_now:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"train_cer={train_cer:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} val_cer={val_cer:.4f} best={best_val_acc:.4f}"
        )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
