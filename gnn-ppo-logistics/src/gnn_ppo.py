"""
GNN encoder + PPO actor-critic, written from scratch in NumPy so every step
(forward pass, gradients, optimiser) is visible and explainable in the viva.

Architecture
------------
  node features X (n x 8) --1 message-passing layer--> node embeddings H (n x 16)
      H = ReLU( X Ws  +  (A_norm X) Wn  + b )      A_norm = mean over neighbours
  pooled:  g = mean of H over ROUTE nodes,  G = mean of H over ALL nodes
  z = [g, G, shipment features]  (16 + 16 + 8 = 40)
  policy head : z -> tanh(64) -> 4 logits   (hold / reroute / expedite / escalate)
  value head  : z -> tanh(64) -> 1 value    (expected future reward)

Because the GNN uses shared weights (Ws, Wn) for every node and only averages over
neighbours, the same network works on graphs of ANY size (tested on unseen 16-node
networks in evaluate.py).
"""
import numpy as np


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


class GNNPolicy:
    def __init__(self, node_f, ship_f, cfg, rng):
        H, M, A = cfg["gnn_hidden"], cfg["mlp_hidden"], cfg["n_actions"]
        self.H = H
        D = 2 * H + ship_f
        init = lambda i, o, s=1.0: rng.normal(0, s / np.sqrt(i), (i, o))
        self.p = dict(
            Ws=init(node_f, H), Wn=init(node_f, H), bg=np.zeros(H),          # GNN layer
            W1=init(D, M), b1=np.zeros(M), W2=init(M, A, 0.01), b2=np.zeros(A),  # policy
            V1=init(D, M), c1=np.zeros(M), V2=init(M, 1), c2=np.zeros(1))        # value
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}   # Adam 1st moment
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}   # Adam 2nd moment
        self.t = 0

    # ----------------------------------------------------------------- forward
    def forward(self, X, A, mask, s):
        p = self.p
        AX = A @ X                                                    # neighbour average
        pre = X @ p["Ws"] + AX @ p["Wn"] + p["bg"]
        Hn = np.maximum(pre, 0)                                       # node embeddings
        msum = mask.sum(1, keepdims=True)
        g = (mask[..., None] * Hn).sum(1) / msum                      # route embedding
        G = Hn.mean(1)                                                # whole-network embedding
        z = np.concatenate([g, G, s], 1)
        a1 = np.tanh(z @ p["W1"] + p["b1"]); logits = a1 @ p["W2"] + p["b2"]
        v1 = np.tanh(z @ p["V1"] + p["c1"]); value = (v1 @ p["V2"] + p["c2"])[:, 0]
        self.cache = (X, AX, pre, Hn, mask, msum, z, a1, v1)
        return logits, value

    # ---------------------------------------------------------------- backward
    def backward(self, dlogits, dvalue):
        p = self.p
        X, AX, pre, Hn, mask, msum, z, a1, v1 = self.cache
        gr = {}
        gr["W2"] = a1.T @ dlogits; gr["b2"] = dlogits.sum(0)
        da = (dlogits @ p["W2"].T) * (1 - a1 ** 2)
        gr["W1"] = z.T @ da; gr["b1"] = da.sum(0)
        dv = dvalue[:, None]
        gr["V2"] = v1.T @ dv; gr["c2"] = dv.sum(0)
        dvv = (dv @ p["V2"].T) * (1 - v1 ** 2)
        gr["V1"] = z.T @ dvv; gr["c1"] = dvv.sum(0)
        dz = da @ p["W1"].T + dvv @ p["V1"].T
        H, n = self.H, Hn.shape[1]
        dHn = mask[..., None] * dz[:, None, :H] / msum[..., None] + dz[:, None, H:2 * H] / n
        dpre = dHn * (pre > 0)
        gr["Ws"] = np.einsum("bnf,bnh->fh", X, dpre)
        gr["Wn"] = np.einsum("bnf,bnh->fh", AX, dpre)
        gr["bg"] = dpre.sum((0, 1))
        return gr

    # --------------------------------------------------------------- optimiser
    def adam_step(self, grads, lr, max_norm, b1=0.9, b2=0.999, eps=1e-8):
        norm = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
        scale = min(1.0, max_norm / (norm + 1e-8))                    # gradient clipping
        self.t += 1
        for k, g in grads.items():
            g = g * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh, vh = self.m[k] / (1 - b1 ** self.t), self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)
        return norm

    def act(self, obs_batch, rng, greedy=False):
        logits, value = self.forward(*obs_batch)
        pi = softmax(logits)
        if greedy:
            a = pi.argmax(1)
        else:
            a = (pi.cumsum(1) > rng.random((len(pi), 1))).argmax(1)
        logp = np.log(pi[np.arange(len(a)), a] + 1e-10)
        return a, logp, value, pi

    def save(self, path):
        np.savez(path, **self.p)

    def load(self, path):
        d = np.load(path)
        self.p = {k: d[k] for k in d.files}


def ppo_update(policy, batch, cfg, lr, rng):
    """One PPO update: `epochs` passes over the rollout in shuffled mini-batches.
    Returns training statistics, including the approximate KL after every epoch."""
    X, A, mask, s, act, logp_old, adv, ret = batch
    N, mb = len(act), cfg["minibatch_size"]
    eps, cv, ce = cfg["clip_epsilon"], cfg["value_coef"], cfg["entropy_coef"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    kl_per_epoch, stats = [], dict(pl=[], vl=[], ent=[], clip=[], gnorm=[])
    for _ in range(cfg["epochs"]):
        idx = rng.permutation(N)
        for st in range(0, N, mb):
            j = idx[st:st + mb]; B = len(j)
            logits, v = policy.forward(X[j], A[j], mask[j], s[j])
            pi = softmax(logits); logpi = np.log(pi + 1e-10)
            logp = logpi[np.arange(B), act[j]]
            ratio = np.exp(logp - logp_old[j])
            a_ = adv[j]
            surr = np.minimum(ratio * a_, np.clip(ratio, 1 - eps, 1 + eps) * a_)
            ent = -(pi * logpi).sum(1)
            # ---- gradients of  L = -surr + cv*(v-ret)^2 - ce*entropy
            active = ~(((a_ > 0) & (ratio > 1 + eps)) | ((a_ < 0) & (ratio < 1 - eps)))
            dlogp = np.where(active, -a_ * ratio, 0.0) / B
            onehot = np.zeros_like(pi); onehot[np.arange(B), act[j]] = 1
            dlogits = dlogp[:, None] * (onehot - pi) + ce / B * pi * (logpi + ent[:, None])
            dv = 2 * cv * (v - ret[j]) / B
            gnorm = policy.adam_step(policy.backward(dlogits, dv), lr, cfg["max_grad_norm"])
            stats["pl"].append(-surr.mean()); stats["vl"].append(((v - ret[j]) ** 2).mean())
            stats["ent"].append(ent.mean()); stats["clip"].append((np.abs(ratio - 1) > eps).mean())
            stats["gnorm"].append(gnorm)
        # approximate KL(old || new) on the whole rollout after this epoch
        logits, _ = policy.forward(X, A, mask, s)
        lp = np.log(softmax(logits)[np.arange(N), act] + 1e-10)
        r = np.exp(lp - logp_old)
        kl_per_epoch.append(float(((r - 1) - np.log(r)).mean()))
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    out["kl_per_epoch"] = kl_per_epoch
    return out


if __name__ == "__main__":
    import os, yaml
    from env import LogisticsEnv, NODE_FEATURES, SHIP_FEATURES, ACTION_NAMES
    with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    rng = np.random.default_rng(42)
    env = LogisticsEnv(cfg, rng=rng)
    policy = GNNPolicy(NODE_FEATURES, SHIP_FEATURES, cfg["model"], rng)
    obs = env.reset()
    batch = (obs["X"][None], obs["A"][None], obs["mask"][None], obs["s"][None])
    a, logp, value, pi = policy.act(batch, rng)
    for name, p in zip(ACTION_NAMES, pi[0]):
        print(f"{name:9s} {p:.3f}")
    print("Chosen action:", ACTION_NAMES[a[0]], "| value estimate:", round(float(value[0]), 3))
    print("Bigger network (16 nodes) works too:",
          policy.forward(*(lambda o: (o["X"][None], o["A"][None], o["mask"][None], o["s"][None]))(
              LogisticsEnv(cfg, n_nodes=16, rng=rng).reset()))[0].shape)
