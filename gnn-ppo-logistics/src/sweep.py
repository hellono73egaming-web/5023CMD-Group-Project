"""
Hyperparameter experiments reported in Section 2.2:
  1. learning-rate sweep   {1e-3, 3e-4, 1e-4}
  2. mini-batch sweep      {64, 256, 512}
  3. KL divergence per PPO epoch (from the main run) -> justifies 10 epochs
  4. training curves of the main run

Run AFTER src/train.py:   python src/sweep.py   -> outputs/*.png, logs/sweep_summary.txt
"""
import json, os, sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # run from project root
from train import train


def summarise(h, tail=5):
    last = h[-tail:]
    return dict(final_return=float(np.mean([r["mean_return"] for r in last])),
                final_on_time=float(np.mean([r["on_time"] for r in last])),
                return_std_over_training=float(np.std([r["mean_return"] for r in h[len(h) // 2:]])),
                max_drop=float(max(0, max(h[i]["mean_return"] - h[j]["mean_return"]
                                          for i in range(len(h)) for j in range(i, len(h))))))


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    ex = cfg["experiments"]
    os.makedirs("outputs", exist_ok=True)
    runs, lines = {}, ["Hyperparameter sweeps", "=" * 80]

    for lr in ex["lr_sweep"]:
        _, h = train(cfg, name=f"lr_{lr:g}", lr=lr, steps=ex["sweep_steps"], quiet=True)
        runs[("lr", lr)] = h
    for mb in ex["minibatch_sweep"]:
        _, h = train(cfg, name=f"mb_{mb}", minibatch=mb, steps=ex["sweep_steps"], quiet=True)
        runs[("mb", mb)] = h

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, (kind, title) in enumerate([("lr", "Learning-rate sweep"), ("mb", "Mini-batch size sweep")]):
        lines.append(f"\n{title} ({ex['sweep_steps']} steps each)")
        lines.append(f"{'setting':>10s} {'final return':>13s} {'final on-time':>14s} "
                     f"{'return std (2nd half)':>22s} {'largest drop':>13s}")
        for (kk, val), h in runs.items():
            if kk != kind: continue
            s = summarise(h)
            lines.append(f"{val:>10g} {s['final_return']:+13.3f} {s['final_on_time']:14.1%} "
                         f"{s['return_std_over_training']:22.3f} {s['max_drop']:13.3f}")
            ax[k].plot([r["steps"] for r in h], [r["mean_return"] for r in h], label=f"{kind}={val:g}")
        ax[k].set_title(title); ax[k].set_xlabel("environment steps"); ax[k].set_ylabel("mean episode reward")
        ax[k].legend(); ax[k].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("outputs/hyperparameter_sweeps.png", dpi=160)

    main = json.load(open("logs/main_history.json"))
    early = [r["kl_per_epoch"] for r in main[:10]]
    kl = np.mean(early, 0)
    lines.append("\nApproximate KL after each PPO epoch (mean of first 10 updates, main run)")
    lines.append("  " + "  ".join(f"e{i + 1}:{v:.4f}" for i, v in enumerate(kl)))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(range(1, len(kl) + 1), kl, "o-", color="#2E6DB4")
    ax[0].set_title("Policy change (approx. KL) vs PPO epoch"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("KL")
    ax[0].grid(alpha=0.3)
    st = [r["steps"] for r in main]
    ax[1].plot(st, [r["mean_return"] for r in main], color="#2E6DB4", label="mean reward")
    ax[1].plot(st, [r["on_time"] for r in main], color="#E07B39", label="on-time rate")
    ax[1].set_title("Training curve (main run)"); ax[1].set_xlabel("environment steps"); ax[1].legend()
    ax[1].grid(alpha=0.3)
    for a in ["hold", "reroute", "expedite", "escalate"]:
        ax[2].plot(st, [r["action_share"][a] for r in main], label=a)
    ax[2].set_title("Action mix during training"); ax[2].set_xlabel("environment steps"); ax[2].legend()
    ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("outputs/training_curves.png", dpi=160)

    open("logs/sweep_summary.txt", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
