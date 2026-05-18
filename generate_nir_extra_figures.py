#!/usr/bin/env python3
from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------- Transformer extra ----------------

def generate_transformer_extras(root: Path):
    from pr_2.pr_2 import AttentionClassifier, RNNClassifier, generate_context_dataset
    from pr_3.pr_3 import MiniTransformer, create_toy_seq2seq_data

    out = make_dir(root / "transformer")
    device = torch.device("cpu")

    # Attention vs RNN extended curves
    x, y = generate_context_dataset(n_samples=6000, seq_len=24, vocab_size=80, seed=42)
    split = int(0.8 * len(x))
    train_x, train_y = x[:split], y[:split]
    test_x, test_y = x[split:], y[split:]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=96, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=96)

    def train_cls(model, epochs=14):
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        ce = nn.CrossEntropyLoss()
        tl, ta, va = [], [], []
        for _ in range(epochs):
            model.train()
            loss_sum, acc_sum, n = 0.0, 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = ce(logits, yb)
                loss.backward()
                opt.step()
                loss_sum += float(loss.item())
                acc_sum += float((logits.argmax(1) == yb).float().mean().item())
                n += 1
            model.eval()
            with torch.no_grad():
                vac, m = 0.0, 0
                for xb, yb in test_loader:
                    logits = model(xb.to(device))
                    vac += float((logits.argmax(1) == yb.to(device)).float().mean().item())
                    m += 1
            tl.append(loss_sum / max(n, 1))
            ta.append(acc_sum / max(n, 1))
            va.append(vac / max(m, 1))
        return np.array(tl), np.array(ta), np.array(va)

    attn = AttentionClassifier(vocab_size=80, d_model=96, num_heads=4, num_classes=2)
    rnn = RNNClassifier(vocab_size=80, d_model=96, num_classes=2)

    atl, ata, ava = train_cls(attn)
    rtl, rta, rva = train_cls(rnn)

    epochs = np.arange(1, len(atl) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, ata, label="Attention train acc", marker="o")
    plt.plot(epochs, rta, label="RNN train acc", marker="s")
    plt.plot(epochs, ava, label="Attention test acc", linestyle="--")
    plt.plot(epochs, rva, label="RNN test acc", linestyle="--")
    plt.title("Attention vs RNN: accuracy trajectories")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.4, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "08_attention_rnn_full_accuracy_trajectories.png", dpi=180)
    plt.close()

    gap = ava - rva
    plt.figure(figsize=(8.6, 4.8))
    plt.plot(epochs, gap, color="#C44E52", marker="o")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Generalization gap: Attention(test) - RNN(test)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy difference")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "09_attention_minus_rnn_gap.png", dpi=180)
    plt.close()

    # MiniTransformer overlays
    bos_id, eos_id = 1, 2
    src_train, tgt_train = create_toy_seq2seq_data(3200, 10, 70, bos_id, eos_id, seed=7)
    src_test, tgt_test = create_toy_seq2seq_data(800, 10, 70, bos_id, eos_id, seed=8)

    def train_mini(cfg, epochs=8):
        model = MiniTransformer(vocab_size=70, d_model=cfg["d_model"], num_heads=cfg["heads"], num_layers=cfg["layers"], ff_dim=cfg["ff_dim"], max_len=12).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        ce = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(src_train, tgt_train), batch_size=64, shuffle=True)
        loss_curve, acc_curve = [], []
        for _ in range(epochs):
            model.train()
            lsum, n = 0.0, 0
            for sb, tb in loader:
                sb, tb = sb.to(device), tb.to(device)
                inp, out = tb[:, :-1], tb[:, 1:]
                opt.zero_grad()
                logits = model(sb, inp)
                loss = ce(logits.reshape(-1, logits.size(-1)), out.reshape(-1))
                loss.backward()
                opt.step()
                lsum += float(loss.item())
                n += 1
            model.eval()
            with torch.no_grad():
                logits = model(src_test.to(device), tgt_test[:, :-1].to(device))
                pred = logits.argmax(-1)
                acc = float((pred == tgt_test[:, 1:].to(device)).float().mean().item())
            loss_curve.append(lsum / max(n, 1))
            acc_curve.append(acc)
        return np.array(loss_curve), np.array(acc_curve)

    small_cfg = {"d_model": 64, "heads": 4, "layers": 2, "ff_dim": 128}
    large_cfg = {"d_model": 96, "heads": 6, "layers": 3, "ff_dim": 192}
    sl, sa = train_mini(small_cfg)
    ll, la = train_mini(large_cfg)
    ep2 = np.arange(1, len(sl) + 1)

    plt.figure(figsize=(8.8, 4.8))
    plt.plot(ep2, sl, marker="o", label="small")
    plt.plot(ep2, ll, marker="s", label="large")
    plt.title("MiniTransformer: train loss overlay")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "10_minitransformer_loss_overlay.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.8, 4.8))
    plt.plot(ep2, sa, marker="o", label="small")
    plt.plot(ep2, la, marker="s", label="large")
    plt.title("MiniTransformer: test token accuracy overlay")
    plt.xlabel("Epoch")
    plt.ylabel("Token accuracy")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "11_minitransformer_accuracy_overlay.png", dpi=180)
    plt.close()

    # Sensitivity (heads in attention model)
    heads_list = [1, 2, 4, 8]
    final_acc = []
    for h in heads_list:
        model = AttentionClassifier(vocab_size=80, d_model=96, num_heads=h, num_classes=2)
        _, _, va = train_cls(model, epochs=8)
        final_acc.append(float(va[-1]))

    plt.figure(figsize=(7.8, 4.8))
    plt.plot(heads_list, final_acc, marker="o", color="#4C72B0")
    plt.title("Attention model sensitivity to number of heads")
    plt.xlabel("Number of heads")
    plt.ylabel("Final test accuracy")
    plt.ylim(0.4, 1.0)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "12_attention_heads_sensitivity.png", dpi=180)
    plt.close()


# ---------------- GAN extra ----------------

class G(nn.Module):
    def __init__(self, zdim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, 64), nn.ReLU(True), nn.Linear(64, 64), nn.ReLU(True), nn.Linear(64, 2)
        )

    def forward(self, z):
        return self.net(z)


class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.LeakyReLU(0.2, True), nn.Linear(64, 64), nn.LeakyReLU(0.2, True), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def sample_real(batch, device):
    centers = torch.tensor(
        [[2 * math.cos(2 * math.pi * k / 8), 2 * math.sin(2 * math.pi * k / 8)] for k in range(8)],
        dtype=torch.float32,
        device=device,
    )
    ids = torch.randint(0, 8, (batch,), device=device)
    return centers[ids] + 0.18 * torch.randn(batch, 2, device=device)


def generate_gan_extras(root: Path):
    out = make_dir(root / "gan")
    device = torch.device("cpu")
    gen, dis = G(16).to(device), D().to(device)
    og = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(dis.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    epochs, steps, batch = 220, 24, 256
    g_curve, d_curve, dr_curve, df_curve = [], [], [], []

    for _ in range(epochs):
        gs = ds = dr = df = 0.0
        for _ in range(steps):
            real = sample_real(batch, device)
            z = torch.randn(batch, 16, device=device)
            fake = gen(z).detach()
            ones = torch.ones(batch, 1, device=device)
            zeros = torch.zeros(batch, 1, device=device)

            pr = dis(real)
            pf = dis(fake)
            ld = bce(pr, ones) + bce(pf, zeros)
            od.zero_grad(); ld.backward(); od.step()

            z = torch.randn(batch, 16, device=device)
            gx = gen(z)
            pg = dis(gx)
            lg = bce(pg, ones)
            og.zero_grad(); lg.backward(); og.step()

            gs += float(lg.item()); ds += float(ld.item())
            dr += float(pr.mean().item()); df += float(pf.mean().item())

        g_curve.append(gs / steps); d_curve.append(ds / steps)
        dr_curve.append(dr / steps); df_curve.append(df / steps)

    ep = np.arange(1, epochs + 1)

    # phase portrait
    plt.figure(figsize=(7.8, 6.4))
    plt.plot(d_curve, g_curve, linewidth=1.5)
    plt.scatter(d_curve[0], g_curve[0], c="green", label="start", s=60)
    plt.scatter(d_curve[-1], g_curve[-1], c="red", label="end", s=60)
    plt.title("GAN training trajectory in loss phase-space")
    plt.xlabel("Discriminator loss")
    plt.ylabel("Generator loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "05_phase_space_g_vs_d_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.8, 4.8))
    plt.plot(ep, np.array(dr_curve) - np.array(df_curve), color="#8172B2")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Discriminator margin: D(real)-D(fake)")
    plt.xlabel("Epoch")
    plt.ylabel("Margin")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "06_discriminator_margin_curve.png", dpi=180)
    plt.close()

    with torch.no_grad():
        real = sample_real(12000, device).cpu().numpy()
        fake = gen(torch.randn(12000, 16, device=device)).cpu().numpy()

    # radial profile
    rr = np.sqrt(real[:, 0] ** 2 + real[:, 1] ** 2)
    rf = np.sqrt(fake[:, 0] ** 2 + fake[:, 1] ** 2)
    bins = np.linspace(0, 4, 45)
    hr, _ = np.histogram(rr, bins=bins, density=True)
    hf, _ = np.histogram(rf, bins=bins, density=True)
    centers = 0.5 * (bins[1:] + bins[:-1])

    plt.figure(figsize=(8.8, 4.8))
    plt.plot(centers, hr, label="real radial density")
    plt.plot(centers, hf, label="generated radial density")
    plt.title("Radial density comparison: real vs generated")
    plt.xlabel("Radius")
    plt.ylabel("Density")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "07_radial_density_real_vs_generated.png", dpi=180)
    plt.close()

    # occupancy difference heatmap
    Hreal, xedges, yedges = np.histogram2d(real[:, 0], real[:, 1], bins=60, range=[[-3.5, 3.5], [-3.5, 3.5]], density=True)
    Hfake, _, _ = np.histogram2d(fake[:, 0], fake[:, 1], bins=[xedges, yedges], density=True)
    diff = Hfake - Hreal

    plt.figure(figsize=(7.4, 6.8))
    vmax = np.percentile(np.abs(diff), 98)
    im = plt.imshow(diff.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax,
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.title("Occupancy difference heatmap (generated - real)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out / "08_occupancy_difference_heatmap.png", dpi=180)
    plt.close()


# ---------------- GNN extra ----------------

class GCN(nn.Module):
    def __init__(self, in_f, hid, out_f, p=0.45):
        super().__init__()
        self.fc1 = nn.Linear(in_f, hid, bias=False)
        self.fc2 = nn.Linear(hid, out_f, bias=False)
        self.p = p

    def forward(self, x, a):
        h = a @ x
        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=self.p, training=self.training)
        h = a @ h
        return self.fc2(h)

    def emb(self, x, a):
        return F.relu(self.fc1(a @ x))


class MLP(nn.Module):
    def __init__(self, in_f, hid, out_f, p=0.45):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_f, hid), nn.ReLU(), nn.Dropout(p), nn.Linear(hid, out_f))

    def forward(self, x):
        return self.net(x)


def make_graph(nc=3, npc=140, fd=24, p_in=0.11, p_out=0.015, seed=42):
    rng = np.random.default_rng(seed)
    n = nc * npc
    labels = np.repeat(np.arange(nc), npc)
    probs = np.full((nc, nc), p_out)
    np.fill_diagonal(probs, p_in)
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < probs[labels[i], labels[j]]:
                A[i, j] = A[j, i] = 1
    centers = rng.normal(0, 1.6, size=(nc, fd)).astype(np.float32)
    X = centers[labels] + rng.normal(0, 0.8, size=(n, fd)).astype(np.float32)
    A2 = A + np.eye(n, dtype=np.float32)
    deg = A2.sum(1)
    d = 1 / np.sqrt(np.maximum(deg, 1e-8))
    Ah = (d[:, None] * A2) * d[None, :]
    return torch.tensor(X), torch.tensor(labels).long(), torch.tensor(Ah), A


def generate_gnn_extras(root: Path):
    out = make_dir(root / "gnn")
    device = torch.device("cpu")
    X, y, Ah, Abin = make_graph()
    n = X.size(0)
    rng = np.random.default_rng(42)
    idx = np.arange(n); rng.shuffle(idx)
    ntr, nval = int(0.6*n), int(0.2*n)
    tr, te = idx[:ntr], idx[ntr+nval:]

    X, y, Ah = X.to(device), y.to(device), Ah.to(device)
    gcn = GCN(X.size(1), 48, 3).to(device)
    mlp = MLP(X.size(1), 48, 3).to(device)
    og = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
    om = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)

    eg, em = 260, 260
    g_acc, m_acc = [], []
    for _ in range(max(eg, em)):
        gcn.train(); og.zero_grad()
        gl = F.cross_entropy(gcn(X, Ah)[tr], y[tr]); gl.backward(); og.step()
        mlp.train(); om.zero_grad()
        ml = F.cross_entropy(mlp(X)[tr], y[tr]); ml.backward(); om.step()

        gcn.eval(); mlp.eval()
        with torch.no_grad():
            pg = gcn(X, Ah)[te].argmax(1)
            pm = mlp(X)[te].argmax(1)
            g_acc.append(float((pg == y[te]).float().mean().item()))
            m_acc.append(float((pm == y[te]).float().mean().item()))

    # confusion matrices
    def confmat(y_true, y_pred, k=3):
        cm = np.zeros((k, k), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        return cm

    with torch.no_grad():
        lg = gcn(X, Ah)[te]
        lm = mlp(X)[te]
    yg = lg.argmax(1).cpu().numpy()
    ym = lm.argmax(1).cpu().numpy()
    yt = y[te].cpu().numpy()

    cmg = confmat(yt, yg)
    cmm = confmat(yt, ym)

    for cm, name in [(cmg, "07_confusion_matrix_gcn.png"), (cmm, "08_confusion_matrix_mlp.png")]:
        plt.figure(figsize=(5.6, 4.8))
        im = plt.imshow(cm, cmap="Blues")
        plt.title(name.replace("_", " ").replace(".png", ""))
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(out / name, dpi=180)
        plt.close()

    # class-wise accuracy
    cls = np.unique(yt)
    gcls, mcls = [], []
    for c in cls:
        mask = yt == c
        gcls.append(float((yg[mask] == yt[mask]).mean()))
        mcls.append(float((ym[mask] == yt[mask]).mean()))

    x = np.arange(len(cls))
    w = 0.35
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - w/2, gcls, w, label="GCN")
    plt.bar(x + w/2, mcls, w, label="MLP")
    plt.xticks(x, [f"Class {c}" for c in cls])
    plt.ylim(0, 1.05)
    plt.title("Class-wise accuracy: GCN vs MLP")
    plt.ylabel("Accuracy")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "09_classwise_accuracy_comparison.png", dpi=180)
    plt.close()

    # degree vs confidence
    deg = Abin.sum(axis=1)
    conf = torch.softmax(lg, dim=1).max(dim=1).values.cpu().numpy()
    deg_te = deg[te]
    plt.figure(figsize=(8.2, 5.0))
    plt.scatter(deg_te, conf, s=14, alpha=0.6)
    z = np.polyfit(deg_te, conf, 1)
    xx = np.linspace(deg_te.min(), deg_te.max(), 100)
    plt.plot(xx, z[0]*xx + z[1], color="#C44E52", linewidth=2, label="linear trend")
    plt.title("Node degree vs GCN confidence (test nodes)")
    plt.xlabel("Node degree")
    plt.ylabel("Max softmax confidence")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "10_degree_vs_confidence_scatter.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    seed_everything(42)
    base = make_dir(Path("report_graphics") / "nir_extra")
    generate_transformer_extras(base)
    generate_gan_extras(base)
    generate_gnn_extras(base)
    print("Saved extra figures to", base.resolve())
