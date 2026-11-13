"""
Simulated regional delivery network for the GNN-PPO logistics agent.

Each episode = one shipment travelling a route through a randomly generated
network graph. The agent makes a decision at each of 3 "legs" of the journey:

    0 = HOLD      keep the current plan
    1 = REROUTE   call the route-planning tool (may fail, driver may ignore it)
    2 = EXPEDITE  switch to a premium carrier (expensive, fixes delay + cold chain)
    3 = ESCALATE  hand the decision to a human supervisor (slow, but reliable)

The four Task 1 uncertainty sources are built in (Section 1.2):
    1.2.1 noisy sensors      -> GPS lateness and cargo temperature are read with noise / drift
    1.2.2 incomplete data    -> supplier delay status is sometimes missing
    1.2.3 human intervention -> the driver may ignore a reroute
    1.2.4 tool failures      -> reroute / expedite API calls can fail
Their strength is re-sampled every episode (domain randomisation), so the policy
cannot over-fit to one fixed setting.
"""
import numpy as np
import networkx as nx

HOLD, REROUTE, EXPEDITE, ESCALATE = 0, 1, 2, 3
ACTION_NAMES = ["hold", "reroute", "expedite", "escalate"]
NODE_FEATURES = 8      # see _observe()
SHIP_FEATURES = 8      # see _observe()


def bayes_delay_posterior(supplier_alert, gps_late, supplier_missing):
    """Task 1 Section 1.3.2 Bayesian update, re-used as an input feature.
    Prior P(D)=0.10, P(alert|D)=0.8, P(alert|~D)=0.15, P(late|D)=0.7, P(late|~D)=0.2."""
    p = 0.10
    if not supplier_missing:                       # missing data -> skip this evidence
        lik_d, lik_n = (0.8, 0.15) if supplier_alert else (0.2, 0.85)
        p = lik_d * p / (lik_d * p + lik_n * (1 - p))
    lik_d, lik_n = (0.7, 0.2) if gps_late else (0.3, 0.8)
    return lik_d * p / (lik_d * p + lik_n * (1 - p))


class LogisticsEnv:
    def __init__(self, cfg, n_nodes=None, rng=None):
        self.e, self.r = cfg["environment"], cfg["reward"]
        self.n = n_nodes or self.e["n_nodes"]
        self.L = self.e["legs_per_episode"]
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        n, rng = self.n, self.rng
        pos = rng.random((n, 2))
        dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):                                   # k-nearest-neighbour roads
            for j in np.argsort(dist[i])[1:self.e["k_neighbours"] + 1]:
                G.add_edge(i, int(j), w=dist[i, j])
        full = nx.Graph()                                    # spanning tree => connected
        full.add_weighted_edges_from((i, j, dist[i, j]) for i in range(n) for j in range(i + 1, n))
        for i, j in nx.minimum_spanning_tree(full).edges():
            G.add_edge(i, j, w=dist[i, j])
        A = nx.to_numpy_array(G, nodelist=range(n), weight=None)
        self.A_norm = A / A.sum(1, keepdims=True)            # mean aggregation for the GNN

        roles = rng.permutation(n)                           # 2 warehouses, 2 suppliers, 2 customers, rest hubs
        self.ntype = np.full(n, 2)                           # 0 warehouse 1 supplier 2 hub 3 customer
        self.ntype[roles[:2]], self.ntype[roles[2:4]], self.ntype[roles[4:6]] = 0, 1, 3
        sup, wh, cus = roles[2], roles[0], roles[4]
        path = nx.shortest_path(G, sup, wh, weight="w") + nx.shortest_path(G, wh, cus, weight="w")[1:]
        self.route = np.zeros(n)
        self.route[path] = 1.0
        self.route_supplier = sup
        return path

    # ----------------------------------------------------------------- episode
    def reset(self):
        rng, e = self.rng, self.e
        # domain randomisation of the Task 1 uncertainty levels
        self.gps_std = rng.uniform(*e["gps_noise_std"])
        self.drift = rng.uniform(*e["sensor_drift"])
        self.p_missing = rng.uniform(*e["supplier_missing_prob"])
        self.p_ignore = rng.uniform(*e["driver_ignore_prob"])
        self.p_fail = rng.uniform(*e["tool_fail_prob"])

        path = self._build_graph()
        weather = rng.random()
        self.congestion = np.clip(rng.beta(2, 5, self.n) + 0.4 * weather, 0, 1)
        self.supplier_delay = {i: rng.random() < rng.uniform(0.1, 0.5) for i in np.where(self.ntype == 1)[0]}
        self.supplier_missing = {i: rng.random() < self.p_missing for i in self.supplier_delay}
        self.obs_congestion = np.clip(self.congestion + rng.normal(0, 0.1, self.n), 0, 1)

        # hidden ground truth: total delay (hours) the shipment will suffer if nothing is done
        self.D = max(0.0, 5 * self.congestion[path].mean()
                     + 3 * self.supplier_delay[self.route_supplier] + rng.normal(0, 0.5) - 1.0)
        self.cold = rng.random() < e["cold_chain_share"]
        self.temp_risk = rng.random() if self.cold else 0.0
        self.t, self.reroutes, self.expedited, self.tool_failed = 0, 0, False, False
        self.stats = dict(escalations=0, tool_failures=0, ignored=0, loop_halt=False)
        return self._observe()

    def _accrued(self):
        return self.D * self.t / self.L          # delay already "visible" on the road

    def _observe(self):
        rng = self.rng
        X = np.zeros((self.n, NODE_FEATURES))
        X[np.arange(self.n), self.ntype] = 1.0                       # node type one-hot (4)
        X[:, 4] = self.obs_congestion                                # noisy congestion reading
        for i in self.supplier_delay:                                 # supplier status (may be missing)
            X[i, 5] = 0.0 if self.supplier_missing[i] else float(self.supplier_delay[i])
            X[i, 6] = float(self.supplier_missing[i])
        X[:, 7] = self.route

        obs_late = self._accrued() + rng.normal(0, self.gps_std)      # noisy GPS lateness
        obs_temp = np.clip(self.temp_risk + self.drift + rng.normal(0, 0.05), 0, 1) if self.cold else 0.0
        s = self.route_supplier
        p_bayes = bayes_delay_posterior(self.supplier_delay[s], obs_late > 0.5, self.supplier_missing[s])
        ship = np.array([self.t / self.L, float(self.cold), obs_temp, obs_late / 5.0,
                         float(self.tool_failed), self.reroutes / self.e["max_repeat_reroutes"],
                         float(self.expedited), p_bayes])
        self.last_obs_late, self.last_obs_temp, self.last_p = obs_late, obs_temp, p_bayes
        return dict(X=X, A=self.A_norm, mask=self.route.copy(), s=ship)

    # ------------------------------------------------------------- dynamics
    def _apply(self, action, reliable=False):
        """Apply hold/reroute/expedite to the remaining (not yet accrued) delay."""
        remaining = self.D - self._accrued()
        if action == REROUTE:
            if not reliable and self.rng.random() < self.p_fail:      # 1.2.4 tool failure
                self.tool_failed = True; self.stats["tool_failures"] += 1; return
            if not reliable and self.rng.random() < self.p_ignore:    # 1.2.3 driver ignores
                self.stats["ignored"] += 1; return
            self.D = self._accrued() + 0.5 * remaining + 0.3          # shorter delay, longer distance
        elif action == EXPEDITE:
            if not reliable and self.rng.random() < self.p_fail:
                self.tool_failed = True; self.stats["tool_failures"] += 1; return
            self.D = self._accrued() + 0.2 * remaining
            self.expedited = True

    def _terminal_reward(self, delay, expedited):
        r = self.r["on_time"] if delay < 0.5 else self.r["delay_per_hour"] * delay
        if self.cold and self.temp_risk > 0.6 and delay > 2.0 and not expedited:
            r += self.r["cold_chain_violation"]
        return r

    def oracle_action(self):
        """Best of hold/reroute/expedite using the TRUE state (simulated supervisor)."""
        rem, acc = self.D - self._accrued(), self._accrued()
        cand = {HOLD: (acc + rem, self.expedited, 0.0),
                REROUTE: (acc + 0.5 * rem + 0.3, self.expedited, self.r["tool_call"]),
                EXPEDITE: (acc + 0.2 * rem, True, self.r["tool_call"] + self.r["expedite_premium"])}
        return max(cand, key=lambda a: self._terminal_reward(cand[a][0], cand[a][1]) + cand[a][2])

    def step(self, action):
        r, rw = 0.0, self.r
        self.tool_failed = False
        if action in (REROUTE, EXPEDITE, ESCALATE):
            r += rw["tool_call"]
        if action == ESCALATE:                                        # human decides, but slowly
            self.stats["escalations"] += 1
            best = self.oracle_action()
            if best != HOLD:
                r += rw["expedite_premium"] if best == EXPEDITE else 0.0
                self._apply(best, reliable=True)
            self.D += 1.0                                             # waiting for the human costs 1 h
        else:
            if action == self.oracle_action():                        # simulated supervisor approval
                r += rw["supervisor_approval"]
            if action == EXPEDITE:
                r += rw["expedite_premium"]
            if action == REROUTE:
                self.reroutes += 1
            self._apply(action)

        self.t += 1
        loop_trap = self.reroutes >= self.e["max_repeat_reroutes"]
        done = self.t >= self.L or loop_trap
        info = {}
        if done:
            if loop_trap:                                             # loop detector halts the agent
                r += rw["loop_trap_penalty"]; self.stats["loop_halt"] = True
            delay = self.D
            r += self._terminal_reward(delay, self.expedited)
            info = dict(delay=delay, on_time=delay < 0.5,
                        violation=self.cold and self.temp_risk > 0.6 and delay > 2.0 and not self.expedited,
                        **self.stats)
            return None, r, True, info
        return self._observe(), r, False, info


if __name__ == "__main__":
    import os, yaml
    with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    print("Bayes check, alert + late:", round(bayes_delay_posterior(True, True, False), 3))  # 0.675
    env = LogisticsEnv(cfg, rng=np.random.default_rng(42))
    for ep in range(5):
        obs, done, total = env.reset(), False, 0.0
        actions = []
        while not done:
            a = int(env.rng.integers(4))
            actions.append(ACTION_NAMES[a])
            obs, r, done, info = env.step(a)
            total += r
        print(f"Ep {ep+1}: {actions} | delay {info['delay']:.2f} h | "
              f"on time {info['on_time']} | reward {total:+.2f}")
