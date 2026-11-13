"""
Evaluate the trained GNN-PPO agent against baselines on
  (a) new episodes from the training distribution (12-node networks), and
  (b) UNSEEN, larger 16-node network layouts (generalisation / sim-overfitting check).

Baselines
  random        : picks any action uniformly
  always-hold   : never intervenes
  threshold     : reroute if GPS shows > 1 h lateness, expedite hot cold-chain cargo
  task1-EU      : the Task 1 expected-utility rule on the Bayesian posterior
                  (hold < 0.267 <= reroute <= 0.8 < expedite)

Run:  python src/evaluate.py        -> logs/evaluation.txt, outputs/evaluation.png, outputs/metrics.json
"""
import json, os, sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # run from project root
from env import LogisticsEnv, NODE_FEATURES, SHIP_FEATURES, HOLD, REROUTE, EXPEDITE
from gnn_ppo import GNNPolicy


def run(policy_fn, cfg, n_nodes, episodes, seed):
    env = LogisticsEnv(cfg, n_nodes=n_nodes, rng=np.random.default_rng(seed))
    res = dict(ret=[], on_time=[], delay=[], violation=[], escalations=[], loop=[], tool_fail=[])
    for _ in range(episodes):
        obs, done, R = env.reset(), False, 0.0
        while not done:
            obs, r, done, info = env.step(policy_fn(obs, env))
            R += r
        res["ret"].append(R); res["on_time"].append(info["on_time"]); res["delay"].append(info["delay"])
        res["violation"].append(info["violation"]); res["escalations"].append(info["escalations"])
        res["loop"].append(info["loop_halt"]); res["tool_fail"].append(info["tool_failures"])
    return {k: float(np.mean(v)) for k, v in res.items()}


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    rng = np.random.default_rng(123)
    pol = GNNPolicy(NODE_FEATURES, SHIP_FEATURES, cfg["model"], rng)
    pol.load("models/main.npz")

    def ppo(obs, env):
        a, *_ = pol.act((obs["X"][None], obs["A"][None], obs["mask"][None], obs["s"][None]), rng, greedy=True)
        return int(a[0])

    def eu_rule(obs, env):            # Task 1, Section 1.3.2
        p = obs["s"][7]
        return HOLD if p < 0.267 else (REROUTE if p <= 0.8 else EXPEDITE)

    def threshold(obs, env):
        if obs["s"][1] and obs["s"][2] > 0.6 and not obs["s"][6]:
            return EXPEDITE
        return REROUTE if obs["s"][3] * 5 > 1.0 else HOLD

    policies = {"GNN-PPO (ours)": ppo, "Task1 EU rule": eu_rule, "Threshold rule": threshold,
                "Always hold": lambda o, e: HOLD, "Random": lambda o, e: int(rng.integers(4))}
    ev = cfg["evaluation"]
    out = {}
    lines = ["Evaluation of trained GNN-PPO agent", "=" * 96]
    for label, n in [("seen (12-node)", cfg["environment"]["n_nodes"]), ("UNSEEN (16-node)", ev["unseen_n_nodes"])]:
        lines.append(f"\nNetwork: {label}, {ev['episodes']} episodes")
        lines.append(f"{'policy':16s} {'return':>8s} {'on-time':>8s} {'avg delay h':>11s} "
                     f"{'cold viol.':>10s} {'escal./ep':>9s} {'loop halt':>9s}")
        out[label] = {}
        for name, fn in policies.items():
            m = run(fn, cfg, n, ev["episodes"], seed=999)
            out[label][name] = m
            lines.append(f"{name:16s} {m['ret']:+8.3f} {m['on_time']:8.1%} {m['delay']:11.2f} "
                         f"{m['violation']:10.1%} {m['escalations']:9.2f} {m['loop']:9.1%}")
    os.makedirs("outputs", exist_ok=True)
    open("logs/evaluation.txt", "w").write("\n".join(lines))
    json.dump(out, open("outputs/metrics.json", "w"), indent=2)
    print("\n".join(lines))

    # ---- evaluation screenshot
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    names = list(policies)
    cols = ["#2E6DB4", "#7FA7D6", "#A9B8C9", "#C9CFD6", "#E2E5E9"]
    for k, (metric, title, fmt) in enumerate([("on_time", "On-time delivery rate", "{:.0%}"),
                                              ("ret", "Mean episode reward", "{:+.2f}"),
                                              ("violation", "Cold-chain violation rate", "{:.1%}")]):
        x = np.arange(len(names)); w = 0.4
        for j, (label, hatch) in enumerate([("seen (12-node)", ""), ("UNSEEN (16-node)", "//")]):
            vals = [out[label][n][metric] for n in names]
            b = ax[k].bar(x + (j - 0.5) * w, vals, w, color=cols, hatch=hatch, edgecolor="#333", linewidth=0.5,
                          label=label)
            ax[k].bar_label(b, labels=[fmt.format(v) for v in vals], fontsize=7, rotation=90, padding=2)
        ax[k].set_xticks(x, names, rotation=25, ha="right", fontsize=8)
        ax[k].set_title(title); ax[k].axhline(0, color="#333", lw=0.6)
        ax[k].margins(y=0.25)
    ax[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("GNN-PPO logistics agent vs baselines (hatched = unseen 16-node networks)", fontsize=11)
    fig.tight_layout(); fig.savefig("outputs/evaluation.png", dpi=160)


if __name__ == "__main__":
    main()
