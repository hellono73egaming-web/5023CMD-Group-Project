"""
Trains the GNN-PPO logistics agent.

Run from the project root (in PyCharm: set the working directory to the project folder):
    python src/train.py                         # main run using config.yaml
    python src/train.py --lr 1e-3 --name lr_1e-3 --steps 150000   # used by sweep.py

Outputs:
    logs/<name>_train_log.txt     human-readable execution log
    logs/<name>_history.json      per-update metrics (for plots)
    models/<name>.npz             trained weights
"""
import argparse, json, os, sys, time
from datetime import datetime
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # run from project root
from env import LogisticsEnv, NODE_FEATURES, SHIP_FEATURES, ACTION_NAMES
from gnn_ppo import GNNPolicy, ppo_update

N_ENVS = 16   # environments stepped in parallel (rollout = N_ENVS x steps_per_env)


def stack(obs_list):
    return (np.stack([o["X"] for o in obs_list]), np.stack([o["A"] for o in obs_list]),
            np.stack([o["mask"] for o in obs_list]), np.stack([o["s"] for o in obs_list]))


def train(cfg, name="main", lr=None, minibatch=None, steps=None, quiet=False):
    pc = dict(cfg["ppo"])
    if lr is not None: pc["learning_rate"] = lr
    if minibatch is not None: pc["minibatch_size"] = minibatch
    if steps is not None: pc["total_steps"] = steps
    os.makedirs("logs", exist_ok=True); os.makedirs("models", exist_ok=True)
    logf = open(f"logs/{name}_train_log.txt", "w", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        logf.write(line + "\n"); logf.flush()
        if not quiet: print(line)

    rng = np.random.default_rng(cfg["seed"])
    envs = [LogisticsEnv(cfg, rng=np.random.default_rng(cfg["seed"] + 1 + i)) for i in range(N_ENVS)]
    policy = GNNPolicy(NODE_FEATURES, SHIP_FEATURES, cfg["model"], rng)
    T = pc["rollout_steps"] // N_ENVS
    n_updates = pc["total_steps"] // pc["rollout_steps"]
    log("=" * 78)
    log(f"GNN-PPO logistics agent | run '{name}'")
    log(f"Hyperparameters: {json.dumps(pc)}")
    log(f"Updates: {n_updates} x {pc['rollout_steps']} steps ({N_ENVS} parallel envs)")

    obs = [e.reset() for e in envs]
    history, ep_returns, ep_infos, cur_ret = [], [], [], np.zeros(N_ENVS)
    t0 = time.time()
    for u in range(n_updates):
        lr_now = pc["learning_rate"] * (1 - u / n_updates) if pc["lr_linear_decay"] else pc["learning_rate"]
        buf = {k: [] for k in ["X", "A", "mask", "s", "act", "logp", "val", "rew", "done"]}
        for _ in range(T):                                   # ---- collect rollout
            X, A, M, S = stack(obs)
            a, logp, val, _ = policy.act((X, A, M, S), rng)
            rews, dones = np.zeros(N_ENVS), np.zeros(N_ENVS)
            for i, e in enumerate(envs):
                o, r, d, info = e.step(int(a[i]))
                rews[i], dones[i] = r, d
                cur_ret[i] += r
                if d:
                    ep_returns.append(cur_ret[i]); ep_infos.append(info); cur_ret[i] = 0
                    o = e.reset()
                obs[i] = o
            for k, v in zip(buf, [X, A, M, S, a, logp, val, rews, dones]):
                buf[k].append(v)
        _, last_val = policy.forward(*stack(obs))
        # ---- generalised advantage estimation (GAE)
        rew, val, done = np.array(buf["rew"]), np.array(buf["val"]), np.array(buf["done"])
        adv, gae = np.zeros_like(rew), 0
        for t in reversed(range(T)):
            nv = last_val if t == T - 1 else val[t + 1]
            delta = rew[t] + pc["gamma"] * nv * (1 - done[t]) - val[t]
            gae = delta + pc["gamma"] * pc["gae_lambda"] * (1 - done[t]) * gae
            adv[t] = gae
        ret = adv + val
        batch = (np.concatenate(buf["X"]), np.concatenate(buf["A"]), np.concatenate(buf["mask"]),
                 np.concatenate(buf["s"]), np.array(buf["act"]).reshape(-1),
                 np.array(buf["logp"]).reshape(-1), adv.reshape(-1), ret.reshape(-1))
        st = ppo_update(policy, batch, pc, lr_now, rng)

        recent, infos = ep_returns[-500:], ep_infos[-500:]
        acts = np.bincount(np.array(buf["act"]).reshape(-1), minlength=4) / pc["rollout_steps"]
        row = dict(update=u + 1, steps=(u + 1) * pc["rollout_steps"], lr=lr_now,
                   mean_return=float(np.mean(recent)),
                   on_time=float(np.mean([i["on_time"] for i in infos])),
                   violation=float(np.mean([i["violation"] for i in infos])),
                   loop_halt=float(np.mean([i["loop_halt"] for i in infos])),
                   action_share={ACTION_NAMES[k]: float(acts[k]) for k in range(4)}, **st)
        history.append(row)
        if (u + 1) % 5 == 0 or u == 0 or u == n_updates - 1:
            log(f"upd {u + 1:3d}/{n_updates} steps {row['steps']:7d} lr {lr_now:.2e} | "
                f"return {row['mean_return']:+.3f} on-time {row['on_time']:.3f} "
                f"cold-viol {row['violation']:.3f} loop {row['loop_halt']:.3f} | "
                f"KL {st['kl_per_epoch'][-1]:.4f} clip {st['clip']:.3f} ent {st['ent']:.3f} "
                f"vloss {st['vl']:.3f}")
    log(f"Final KL by epoch (last update): {[round(k, 4) for k in history[-1]['kl_per_epoch']]}")
    log(f"Training time {time.time() - t0:.1f}s")
    policy.save(f"models/{name}.npz")
    json.dump(history, open(f"logs/{name}_history.json", "w"), indent=1)
    log(f"Saved models/{name}.npz and logs/{name}_history.json")
    return policy, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--name", default="main")
    ap.add_argument("--lr", type=float); ap.add_argument("--minibatch", type=int)
    ap.add_argument("--steps", type=int)
    a = ap.parse_args()
    train(yaml.safe_load(open(a.config)), a.name, a.lr, a.minibatch, a.steps)
