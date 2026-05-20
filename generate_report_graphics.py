#!/usr/bin/env python3
"""Генерация графиков и иллюстраций для курсового отчета.

Покрываем 3 раздела:
1) Трансформеры
2) GAN
3) Графовые нейросети

Скрипт не изменяет существующие практики, а только создает материалы в новой директории.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Общие утилиты
# ----------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не удалось загрузить модуль: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Для генерации PNG нужен matplotlib. Установи: pip install matplotlib"
        ) from exc


def make_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    shutil.copy2(src, dst)
    return True


def build_docx_pack(root: Path) -> Path:
    """Собирает финальный набор картинок для прямой вставки в отчет."""
    docx_dir = make_dir(root / "for_docx")

    by_section = {
        "1_transformer": make_dir(docx_dir / "1_transformer"),
        "2_gan": make_dir(docx_dir / "2_gan"),
        "3_gnn": make_dir(docx_dir / "3_gnn"),
    }

    transformer_items = [
        ("01_positional_encoding_heatmap.png", "Тепловая карта позиционного кодирования"),
        ("02_attention_weights_head0.png", "Матрица весов самовнимания"),
        ("03_attention_vs_rnn_train_loss.png", "Сравнение Attention и RNN по функции потерь на обучении"),
        ("04_attention_vs_rnn_test_accuracy.png", "Сравнение Attention и RNN по точности на тесте"),
        ("05_minitransformer_loss_small.png", "MiniTransformer small: функция потерь"),
        ("06_minitransformer_acc_small.png", "MiniTransformer small: точность токенов"),
        ("07_minitransformer_final_comparison.png", "Финальное сравнение конфигураций MiniTransformer"),
    ]

    gan_distribution = sorted((root / "gan").glob("01_distribution_epoch_*.png"))
    gan_items: List[Tuple[str, str]] = [
        (p.name, f"Распределение реальных и сгенерированных точек, {p.stem.replace('01_distribution_', '').replace('_', ' ')}")
        for p in gan_distribution
    ]
    gan_items.extend(
        [
            ("02_gan_losses.png", "Кривые потерь GAN"),
            ("03_discriminator_confidence.png", "Динамика уверенности дискриминатора"),
            ("04_generated_density_hexbin.png", "Плотность сгенерированных точек"),
        ]
    )

    gnn_items = [
        ("01_gcn_vs_mlp_train_loss.png", "GCN vs MLP: функция потерь на обучении"),
        ("02_gcn_vs_mlp_test_accuracy.png", "GCN vs MLP: точность на тесте"),
        ("03_final_accuracy_bar.png", "Финальная точность методов"),
        ("04_gcn_embeddings_pca.png", "PCA эмбеддингов GCN"),
        ("05_graph_structure_subgraph.png", "Визуализация подграфа"),
        ("06_degree_distribution.png", "Распределение степеней узлов"),
    ]

    sections = [
        ("1", "Трансформер", root / "transformer", transformer_items, by_section["1_transformer"]),
        ("2", "Генеративно-состязательная сеть (GAN)", root / "gan", gan_items, by_section["2_gan"]),
        ("3", "Графовая нейросеть", root / "gnn", gnn_items, by_section["3_gnn"]),
    ]

    lines: List[str] = ["# План вставки графиков в отчет", ""]

    for section_num, section_title, source_dir, items, section_out_dir in sections:
        lines.append(f"## Раздел {section_num}. {section_title}")
        lines.append("")

        img_idx = 1
        for filename, caption in items:
            src = source_dir / filename
            target_name = f"{section_num}_{img_idx:02d}_{filename}"
            dst_flat = docx_dir / target_name
            dst_section = section_out_dir / target_name

            copied_main = copy_if_exists(src, dst_flat)
            copied_section = copy_if_exists(src, dst_section)

            if copied_main and copied_section:
                lines.append(f"{img_idx}. `{target_name}` — {caption}")
                img_idx += 1

        lines.append("")

    section4_source = Path("market_nir/artifacts/architecture_comparison/plots")
    section4_dir = make_dir(docx_dir / "4_market_architecture")
    section4_items = [
        ("4_00_market_experiment_pipeline.png", "Конвейер эксперимента раздела 4"),
        ("4_09_split_and_label_distribution.png", "Размеры обучающего, валидационного и тестового разбиений и распределение классов"),
        ("4_11_market_return_context.png", "Накопленная будущая доходность ret_h по тикерам"),
        ("4_10_feature_group_counts.png", "Группы признаков market-only датасета"),
        ("4_12_architecture_assumptions.png", "Сравниваемые архитектуры и их индуктивные предположения"),
        ("4_13_training_val_balanced_accuracy.png", "Сбалансированная accuracy нейросетевых моделей на валидации"),
        ("4_14_training_loss_curves.png", "Функция потерь нейросетевых моделей на обучении"),
        ("4_01_architecture_quality_metrics.png", "Сравнение архитектур по сбалансированной accuracy, macro-F1 и hit-rate top-10% сигналов"),
        ("4_08_metrics_heatmap.png", "Тепловая карта итоговых метрик"),
        ("4_02_quality_vs_train_time.png", "Компромисс качества и времени обучения"),
        ("4_03_inference_latency.png", "Скорость инференса разных архитектур"),
        ("4_04_parameter_complexity.png", "Параметрическая сложность моделей"),
        ("4_18_quality_efficiency_radar.png", "Интегральный профиль качества и эффективности"),
        ("4_15_score_distributions.png", "Распределение score по моделям"),
        ("4_06_score_vs_future_return.png", "Связь score модели и будущей доходности"),
        ("4_16_top10_pnl_by_model.png", "PnL top-10% наиболее уверенных сигналов"),
        ("4_07_equity_curves_top10.png", "Кривые капитала для top-10% сигналов"),
        ("4_05_best_model_confusion_matrix.png", "Матрица ошибок лучшей модели"),
        ("4_17_best_model_rolling_error.png", "Скользящая доля ошибок лучшей модели"),
    ]
    if section4_source.exists():
        lines.append("## Раздел 4. Влияние архитектуры модели на прогнозирование рыночных временных рядов")
        lines.append("")
        lines.append("Графики лежат в `4_market_architecture/`.")
        lines.append("")
        img_idx = 1
        for filename, caption in section4_items:
            src = section4_source / filename
            dst = section4_dir / filename
            if copy_if_exists(src, dst):
                lines.append(f"{img_idx}. `4_market_architecture/{filename}` — {caption}")
                img_idx += 1
        lines.append("")

    (docx_dir / "insert_plan.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return docx_dir


# ----------------------------
# Раздел 1: Трансформеры
# ----------------------------


@dataclass
class Curve:
    train_loss: List[float]
    train_acc: List[float]
    test_acc: List[float]


def train_classifier_with_curves(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Curve:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    train_loss_curve: List[float] = []
    train_acc_curve: List[float] = []
    test_acc_curve: List[float] = []

    for _ in range(epochs):
        model.train()
        loss_sum = 0.0
        acc_sum = 0.0
        steps = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            acc_sum += float((logits.argmax(dim=1) == yb).float().mean().item())
            steps += 1

        model.eval()
        test_acc_sum = 0.0
        test_steps = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                test_acc_sum += float((logits.argmax(dim=1) == yb).float().mean().item())
                test_steps += 1

        train_loss_curve.append(loss_sum / max(steps, 1))
        train_acc_curve.append(acc_sum / max(steps, 1))
        test_acc_curve.append(test_acc_sum / max(test_steps, 1))

    return Curve(train_loss=train_loss_curve, train_acc=train_acc_curve, test_acc=test_acc_curve)


def run_transformer_section(root: Path, quick: bool, device: torch.device) -> Dict[str, object]:
    plt = require_matplotlib()

    pr2 = load_module(Path("pr_2/pr_2.py"), "pr2_module")
    pr3 = load_module(Path("pr_3/pr_3.py"), "pr3_module")

    out_dir = make_dir(root / "transformer")

    # 1) Тепловая карта позиционного кодирования
    d_model = 64
    pe_model = pr3.PositionalEncoding(d_model=d_model, max_len=96)
    pe = pe_model.pe[0, :96, :d_model].detach().cpu().numpy()

    plt.figure(figsize=(10, 5))
    im = plt.imshow(pe.T, aspect="auto", cmap="coolwarm")
    plt.title("Тепловая карта позиционного кодирования")
    plt.xlabel("Позиция токена")
    plt.ylabel("Размерность эмбеддинга")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_dir / "01_positional_encoding_heatmap.png", dpi=170)
    plt.close()

    # 2) Карта весов внимания (одна голова)
    seq_len = 24
    d_attn = 96
    heads = 4
    attn = pr2.MultiHeadAttentionFromScratch(d_model=d_attn, num_heads=heads, dropout=0.0)
    attn.eval()

    x = torch.randn(1, seq_len, d_attn)
    with torch.no_grad():
        q = attn._split_heads(attn.q_proj(x))
        k = attn._split_heads(attn.k_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        weights = torch.softmax(scores, dim=-1)[0, 0].cpu().numpy()

    plt.figure(figsize=(6.6, 5.8))
    im = plt.imshow(weights, cmap="viridis")
    plt.title("Веса самовнимания (голова 0)")
    plt.xlabel("Позиция ключа")
    plt.ylabel("Позиция запроса")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_dir / "02_attention_weights_head0.png", dpi=170)
    plt.close()

    # 3) Attention vs RNN на synthetic context task
    n_samples = 2500 if quick else 5000
    epochs = 7 if quick else 12
    batch_size = 64
    vocab_size = 80

    x_data, y_data = pr2.generate_context_dataset(
        n_samples=n_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=42,
    )

    split = int(0.8 * len(x_data))
    train_ds = torch.utils.data.TensorDataset(x_data[:split], y_data[:split])
    test_ds = torch.utils.data.TensorDataset(x_data[split:], y_data[split:])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

    attn_model = pr2.AttentionClassifier(vocab_size=vocab_size, d_model=d_attn, num_heads=heads, num_classes=2)
    rnn_model = pr2.RNNClassifier(vocab_size=vocab_size, d_model=d_attn, num_classes=2)

    attn_curve = train_classifier_with_curves(attn_model, train_loader, test_loader, device, epochs=epochs, lr=3e-3)
    rnn_curve = train_classifier_with_curves(rnn_model, train_loader, test_loader, device, epochs=epochs, lr=3e-3)

    xs = np.arange(1, epochs + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(xs, attn_curve.train_loss, marker="o", label="Attention")
    plt.plot(xs, rnn_curve.train_loss, marker="s", label="RNN")
    plt.title("Attention и RNN: функция потерь на обучении")
    plt.xlabel("Эпоха")
    plt.ylabel("Потери")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "03_attention_vs_rnn_train_loss.png", dpi=170)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(xs, attn_curve.test_acc, marker="o", label="Attention")
    plt.plot(xs, rnn_curve.test_acc, marker="s", label="RNN")
    plt.title("Attention и RNN: точность на тесте")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность")
    plt.ylim(0.45, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "04_attention_vs_rnn_test_accuracy.png", dpi=170)
    plt.close()

    # 4) Мини-трансформер: сравнение конфигураций
    n_train = 1800 if quick else 3600
    n_test = 400 if quick else 900
    mini_epochs = 5 if quick else 8

    bos_id, eos_id = 1, 2
    src_train, tgt_train = pr3.create_toy_seq2seq_data(
        n_samples=n_train,
        seq_len=10,
        vocab_size=70,
        bos_id=bos_id,
        eos_id=eos_id,
        seed=7,
    )
    src_test, tgt_test = pr3.create_toy_seq2seq_data(
        n_samples=n_test,
        seq_len=10,
        vocab_size=70,
        bos_id=bos_id,
        eos_id=eos_id,
        seed=8,
    )

    configs = [
        {"name": "small", "layers": 2, "d_model": 64, "heads": 4, "ff_dim": 128},
        {"name": "large", "layers": 3, "d_model": 96, "heads": 6, "ff_dim": 192},
    ]

    transformer_results: Dict[str, Dict[str, List[float] | float]] = {}
    criterion = nn.CrossEntropyLoss()

    for cfg in configs:
        model = pr3.MiniTransformer(
            vocab_size=70,
            d_model=cfg["d_model"],
            num_heads=cfg["heads"],
            num_layers=cfg["layers"],
            ff_dim=cfg["ff_dim"],
            max_len=12,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(src_train, tgt_train),
            batch_size=64,
            shuffle=True,
        )

        loss_curve: List[float] = []
        acc_curve: List[float] = []

        for _ in range(mini_epochs):
            model.train()
            loss_sum = 0.0
            steps = 0
            for src_b, tgt_b in train_loader:
                src_b = src_b.to(device)
                tgt_b = tgt_b.to(device)

                inp = tgt_b[:, :-1]
                out_true = tgt_b[:, 1:]

                optimizer.zero_grad()
                logits = model(src_b, inp)
                loss = criterion(logits.reshape(-1, logits.size(-1)), out_true.reshape(-1))
                loss.backward()
                optimizer.step()

                loss_sum += float(loss.item())
                steps += 1

            model.eval()
            with torch.no_grad():
                logits = model(src_test.to(device), tgt_test[:, :-1].to(device))
                preds = logits.argmax(dim=-1)
                target = tgt_test[:, 1:].to(device)
                token_acc = float((preds == target).float().mean().item())

            loss_curve.append(loss_sum / max(steps, 1))
            acc_curve.append(token_acc)

        transformer_results[cfg["name"]] = {
            "train_loss": loss_curve,
            "test_token_acc": acc_curve,
            "final_acc": acc_curve[-1],
        }

        xs_cfg = np.arange(1, mini_epochs + 1)
        plt.figure(figsize=(8.2, 4.8))
        plt.plot(xs_cfg, loss_curve, marker="o")
        plt.title(
            f"MiniTransformer: потери на обучении ({cfg['name']}: L={cfg['layers']}, d={cfg['d_model']}, h={cfg['heads']})"
        )
        plt.xlabel("Эпоха")
        plt.ylabel("Потери")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"05_minitransformer_loss_{cfg['name']}.png", dpi=170)
        plt.close()

        plt.figure(figsize=(8.2, 4.8))
        plt.plot(xs_cfg, acc_curve, marker="o")
        plt.title(
            f"MiniTransformer: точность токенов на тесте ({cfg['name']}: L={cfg['layers']}, d={cfg['d_model']}, h={cfg['heads']})"
        )
        plt.xlabel("Эпоха")
        plt.ylabel("Точность токенов")
        plt.ylim(0.0, 1.0)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"06_minitransformer_acc_{cfg['name']}.png", dpi=170)
        plt.close()

    labels = list(transformer_results.keys())
    finals = [float(transformer_results[k]["final_acc"]) for k in labels]
    plt.figure(figsize=(7, 4.8))
    bars = plt.bar(labels, finals, color=["#4C72B0", "#55A868"])
    plt.title("MiniTransformer: итоговая точность токенов")
    plt.ylabel("Точность")
    plt.ylim(0.0, 1.0)
    for bar, value in zip(bars, finals):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(out_dir / "07_minitransformer_final_comparison.png", dpi=170)
    plt.close()

    return {
        "attention_final_test_acc": float(attn_curve.test_acc[-1]),
        "rnn_final_test_acc": float(rnn_curve.test_acc[-1]),
        "mini_transformer": transformer_results,
    }


# ----------------------------
# Раздел 2: GAN (synthetic)
# ----------------------------


class SimpleGANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SimpleGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



def sample_real_2d(batch_size: int, device: torch.device) -> torch.Tensor:
    """Смесь 8 гауссиан по кругу."""
    centers = []
    for k in range(8):
        angle = 2 * math.pi * k / 8
        centers.append([2.0 * math.cos(angle), 2.0 * math.sin(angle)])
    centers_t = torch.tensor(centers, dtype=torch.float32, device=device)

    ids = torch.randint(0, 8, (batch_size,), device=device)
    means = centers_t[ids]
    noise = 0.18 * torch.randn(batch_size, 2, device=device)
    return means + noise


def run_gan_section(root: Path, quick: bool, device: torch.device) -> Dict[str, object]:
    plt = require_matplotlib()
    out_dir = make_dir(root / "gan")

    latent_dim = 16
    epochs = 140 if quick else 280
    steps_per_epoch = 18 if quick else 30
    batch_size = 256

    G = SimpleGANGenerator(latent_dim=latent_dim).to(device)
    D = SimpleGANDiscriminator().to(device)

    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    g_curve: List[float] = []
    d_curve: List[float] = []
    d_real_curve: List[float] = []
    d_fake_curve: List[float] = []

    snapshot_epochs = {1, max(1, epochs // 3), max(1, 2 * epochs // 3), epochs}

    for epoch in range(1, epochs + 1):
        g_epoch = 0.0
        d_epoch = 0.0
        d_real_epoch = 0.0
        d_fake_epoch = 0.0

        for _ in range(steps_per_epoch):
            real = sample_real_2d(batch_size, device)
            z = torch.randn(batch_size, latent_dim, device=device)
            fake = G(z).detach()

            ones = torch.ones(batch_size, 1, device=device)
            zeros = torch.zeros(batch_size, 1, device=device)

            d_real = D(real)
            d_fake = D(fake)
            loss_d = criterion(d_real, ones) + criterion(d_fake, zeros)

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            z = torch.randn(batch_size, latent_dim, device=device)
            gen = G(z)
            pred = D(gen)
            loss_g = criterion(pred, ones)

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            g_epoch += float(loss_g.item())
            d_epoch += float(loss_d.item())
            d_real_epoch += float(d_real.mean().item())
            d_fake_epoch += float(d_fake.mean().item())

        g_curve.append(g_epoch / steps_per_epoch)
        d_curve.append(d_epoch / steps_per_epoch)
        d_real_curve.append(d_real_epoch / steps_per_epoch)
        d_fake_curve.append(d_fake_epoch / steps_per_epoch)

        if epoch in snapshot_epochs:
            with torch.no_grad():
                real_vis = sample_real_2d(1800, device).cpu().numpy()
                fake_vis = G(torch.randn(1800, latent_dim, device=device)).cpu().numpy()

            plt.figure(figsize=(7.5, 7))
            plt.scatter(real_vis[:, 0], real_vis[:, 1], s=8, alpha=0.35, label="Реальные", c="#4C72B0")
            plt.scatter(fake_vis[:, 0], fake_vis[:, 1], s=8, alpha=0.35, label="Сгенерированные", c="#DD8452")
            plt.title(f"Двумерное распределение GAN (эпоха {epoch})")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.legend(loc="upper right")
            plt.axis("equal")
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(out_dir / f"01_distribution_epoch_{epoch:03d}.png", dpi=170)
            plt.close()

    xs = np.arange(1, epochs + 1)
    plt.figure(figsize=(9.2, 5.2))
    plt.plot(xs, d_curve, label="Потери дискриминатора")
    plt.plot(xs, g_curve, label="Потери генератора")
    plt.title("Функции потерь GAN при обучении")
    plt.xlabel("Эпоха")
    plt.ylabel("Потери")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "02_gan_losses.png", dpi=170)
    plt.close()

    plt.figure(figsize=(9.2, 5.2))
    plt.plot(xs, d_real_curve, label="D(реальные)")
    plt.plot(xs, d_fake_curve, label="D(сгенерированные)")
    plt.title("Динамика уверенности дискриминатора")
    plt.xlabel("Эпоха")
    plt.ylabel("Средняя вероятность")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "03_discriminator_confidence.png", dpi=170)
    plt.close()

    with torch.no_grad():
        final_fake = G(torch.randn(8000, latent_dim, device=device)).cpu().numpy()

    plt.figure(figsize=(8, 6.8))
    plt.hexbin(final_fake[:, 0], final_fake[:, 1], gridsize=45, cmap="magma", mincnt=1)
    plt.title("Плотность сгенерированных точек (финальная эпоха)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    cb = plt.colorbar()
    cb.set_label("Число точек")
    plt.tight_layout()
    plt.savefig(out_dir / "04_generated_density_hexbin.png", dpi=170)
    plt.close()

    return {
        "final_generator_loss": float(g_curve[-1]),
        "final_discriminator_loss": float(d_curve[-1]),
        "final_d_real": float(d_real_curve[-1]),
        "final_d_fake": float(d_fake_curve[-1]),
    }


# ----------------------------
# Раздел 3: Графовые сети (synthetic GCN)
# ----------------------------


class SyntheticGCN(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, out_features, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = a_hat @ x
        h = self.fc1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = a_hat @ h
        out = self.fc2(h)
        return out

    def embeddings(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = a_hat @ x
        h = self.fc1(h)
        return F.relu(h)


class MLPBaseline(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_synthetic_graph(
    n_classes: int = 3,
    nodes_per_class: int = 120,
    feature_dim: int = 24,
    p_in: float = 0.10,
    p_out: float = 0.012,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Создаем SBM-граф + признаки узлов + метки классов."""
    rng = np.random.default_rng(seed)

    n_nodes = n_classes * nodes_per_class
    labels = np.repeat(np.arange(n_classes), nodes_per_class)

    probs = np.full((n_classes, n_classes), p_out, dtype=np.float32)
    np.fill_diagonal(probs, p_in)

    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        ci = labels[i]
        for j in range(i + 1, n_nodes):
            cj = labels[j]
            if rng.random() < probs[ci, cj]:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    # Признаки: классовый центр + шум
    centers = rng.normal(loc=0.0, scale=1.6, size=(n_classes, feature_dim)).astype(np.float32)
    x = centers[labels] + rng.normal(loc=0.0, scale=0.8, size=(n_nodes, feature_dim)).astype(np.float32)

    # Нормализация adjacency с self-loop: A_hat = D^{-1/2}(A+I)D^{-1/2}
    a = adj + np.eye(n_nodes, dtype=np.float32)
    deg = a.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
    a_hat = (d_inv_sqrt[:, None] * a) * d_inv_sqrt[None, :]

    return torch.from_numpy(x), torch.from_numpy(labels).long(), torch.from_numpy(a_hat)


def split_masks(n_nodes: int, train_ratio: float = 0.6, val_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_nodes)
    rng.shuffle(idx)

    n_train = int(n_nodes * train_ratio)
    n_val = int(n_nodes * val_ratio)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    return train_idx, val_idx, test_idx


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


def pca_2d(x: torch.Tensor) -> np.ndarray:
    x0 = x - x.mean(dim=0, keepdim=True)
    # Возвращает U, S, V: координаты = x @ V[:, :2]
    _, _, v = torch.pca_lowrank(x0, q=2)
    coords = x0 @ v[:, :2]
    return coords.detach().cpu().numpy()


def run_gnn_section(root: Path, quick: bool, device: torch.device) -> Dict[str, object]:
    plt = require_matplotlib()

    try:
        import networkx as nx
    except ModuleNotFoundError as exc:
        raise SystemExit("Для графовой визуализации нужен networkx: pip install networkx") from exc

    out_dir = make_dir(root / "gnn")

    x, y, a_hat = build_synthetic_graph(
        n_classes=3,
        nodes_per_class=90 if quick else 140,
        feature_dim=24,
        p_in=0.11,
        p_out=0.015,
        seed=42,
    )

    n_nodes = x.size(0)
    train_idx, val_idx, test_idx = split_masks(n_nodes, seed=42)

    x = x.to(device)
    y = y.to(device)
    a_hat = a_hat.to(device)

    gcn = SyntheticGCN(in_features=x.size(1), hidden=48, out_features=3, dropout=0.45).to(device)
    mlp = MLPBaseline(in_features=x.size(1), hidden=48, out_features=3, dropout=0.45).to(device)

    epochs = 160 if quick else 260
    opt_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)

    gcn_loss_curve: List[float] = []
    gcn_test_curve: List[float] = []
    mlp_loss_curve: List[float] = []
    mlp_test_curve: List[float] = []

    for _ in range(epochs):
        # GCN step
        gcn.train()
        opt_gcn.zero_grad()
        gcn_out = gcn(x, a_hat)
        gcn_loss = F.cross_entropy(gcn_out[train_idx], y[train_idx])
        gcn_loss.backward()
        opt_gcn.step()

        gcn.eval()
        with torch.no_grad():
            test_logits = gcn(x, a_hat)[test_idx]
            test_acc = accuracy_from_logits(test_logits, y[test_idx])

        gcn_loss_curve.append(float(gcn_loss.item()))
        gcn_test_curve.append(test_acc)

        # MLP baseline step
        mlp.train()
        opt_mlp.zero_grad()
        mlp_out = mlp(x)
        mlp_loss = F.cross_entropy(mlp_out[train_idx], y[train_idx])
        mlp_loss.backward()
        opt_mlp.step()

        mlp.eval()
        with torch.no_grad():
            test_logits = mlp(x)[test_idx]
            test_acc = accuracy_from_logits(test_logits, y[test_idx])

        mlp_loss_curve.append(float(mlp_loss.item()))
        mlp_test_curve.append(test_acc)

    xs = np.arange(1, epochs + 1)
    plt.figure(figsize=(9.2, 5.2))
    plt.plot(xs, gcn_loss_curve, label="GCN")
    plt.plot(xs, mlp_loss_curve, label="MLP baseline")
    plt.title("Классификация узлов: функция потерь на обучении")
    plt.xlabel("Эпоха")
    plt.ylabel("Потери")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "01_gcn_vs_mlp_train_loss.png", dpi=170)
    plt.close()

    plt.figure(figsize=(9.2, 5.2))
    plt.plot(xs, gcn_test_curve, label="GCN")
    plt.plot(xs, mlp_test_curve, label="MLP baseline")
    plt.title("Классификация узлов: точность на тесте")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "02_gcn_vs_mlp_test_accuracy.png", dpi=170)
    plt.close()

    final_gcn = gcn_test_curve[-1]
    final_mlp = mlp_test_curve[-1]
    plt.figure(figsize=(6.6, 4.8))
    bars = plt.bar(["GCN", "MLP"], [final_gcn, final_mlp], color=["#4C72B0", "#C44E52"])
    plt.title("Сравнение итоговой точности")
    plt.ylabel("Точность")
    plt.ylim(0.0, 1.0)
    for bar, value in zip(bars, [final_gcn, final_mlp]):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(out_dir / "03_final_accuracy_bar.png", dpi=170)
    plt.close()

    # PCA визуализация эмбеддингов GCN
    gcn.eval()
    with torch.no_grad():
        emb = gcn.embeddings(x, a_hat).detach().cpu()
    coords = pca_2d(emb)
    labels_np = y.detach().cpu().numpy()

    plt.figure(figsize=(7.2, 6.0))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels_np, cmap="tab10", s=12, alpha=0.85)
    plt.title("Эмбеддинги узлов GCN (PCA)")
    plt.xlabel("Главная компонента 1")
    plt.ylabel("Главная компонента 2")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(out_dir / "04_gcn_embeddings_pca.png", dpi=170)
    plt.close()

    # Визуализация структуры графа (подграф)
    a_cpu = a_hat.detach().cpu().numpy()
    # Восстанавливаем бинарную матрицу ребер примерно через порог
    # (используем ненулевые связи исходной структуры после нормализации)
    threshold = float(np.percentile(a_cpu[a_cpu > 0], 55))
    adj_bin = (a_cpu > threshold).astype(np.int32)
    np.fill_diagonal(adj_bin, 0)

    g = nx.from_numpy_array(adj_bin)
    subset = list(range(min(120, n_nodes)))
    sg = g.subgraph(subset).copy()

    pos = nx.spring_layout(sg, seed=42)
    node_colors = [int(labels_np[n]) for n in sg.nodes()]

    plt.figure(figsize=(9.2, 7.0))
    nx.draw_networkx_edges(sg, pos, width=0.4, alpha=0.25)
    nx.draw_networkx_nodes(
        sg,
        pos,
        node_color=node_colors,
        node_size=52,
        cmap=plt.cm.tab10,
        alpha=0.95,
    )
    plt.title("Структура синтетического графа (подграф)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "05_graph_structure_subgraph.png", dpi=170)
    plt.close()

    degrees = np.array([d for _, d in sg.degree()], dtype=np.int32)
    plt.figure(figsize=(7.5, 4.8))
    plt.hist(degrees, bins=18, color="#55A868", edgecolor="white")
    plt.title("Распределение степеней узлов (подграф)")
    plt.xlabel("Степень узла")
    plt.ylabel("Число узлов")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "06_degree_distribution.png", dpi=170)
    plt.close()

    return {
        "gcn_final_test_acc": float(final_gcn),
        "mlp_final_test_acc": float(final_mlp),
        "num_nodes": int(n_nodes),
    }


# ----------------------------
# Main
# ----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генератор графиков для курсовой")
    parser.add_argument("--out-dir", type=str, default="report_graphics", help="Куда сохранять изображения")
    parser.add_argument("--quick", action="store_true", help="Быстрый режим (меньше эпох/данных)")
    parser.add_argument(
        "--skip-docx-pack",
        action="store_true",
        help="Не собирать папку for_docx с нумерацией и планом вставки",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_root = make_dir(Path(args.out_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary: Dict[str, object] = {
        "device": str(device),
        "quick_mode": bool(args.quick),
        "seed": int(args.seed),
    }

    print(f"[1/3] Генерация графиков по трансформерам -> {out_root / 'transformer'}")
    summary["transformer"] = run_transformer_section(out_root, quick=args.quick, device=device)

    print(f"[2/3] Генерация графиков по GAN -> {out_root / 'gan'}")
    summary["gan"] = run_gan_section(out_root, quick=args.quick, device=device)

    print(f"[3/3] Генерация графиков по графовым сетям -> {out_root / 'gnn'}")
    summary["gnn"] = run_gnn_section(out_root, quick=args.quick, device=device)

    summary_path = out_root / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    docx_pack_path = None
    if not args.skip_docx_pack:
        docx_pack_path = build_docx_pack(out_root)

    print("\nГотово.")
    print(f"Изображения: {out_root.resolve()}")
    print(f"Метрики: {summary_path.resolve()}")
    if docx_pack_path is not None:
        print(f"DOCX-пакет: {docx_pack_path.resolve()}")
        print(f"План вставки: {(docx_pack_path / 'insert_plan.md').resolve()}")


if __name__ == "__main__":
    main()
