import numpy as np, networkx as nx

def hk_step(x, adj, U, sig):
    xn = x.copy()
    for i in U:
        nb = [j for j in adj[i] if abs(x[i]-x[j]) <= sig[i]] + [i]
        xn[i] = np.mean(x[nb])
    return xn

def V(x, edges, sigU):
    return sum(min((x[i]-x[j])**2, sigU**2) for i, j in edges)

def dissip(x, xn, adj, U, sig):
    # delta^T (D_U + A_UU) delta, effective graph at time t (incl. self-loops)
    idx = {u: k for k, u in enumerate(U)}
    m = len(U); M = np.zeros((m, m))
    for i in U:
        nb = [j for j in adj[i] if abs(x[i]-x[j]) <= sig[i]]
        M[idx[i], idx[i]] += len(nb) + 1 + 1   # D_ii (incl. loop) + A_ii (loop)
        for j in nb:
            if j in idx: M[idx[i], idx[j]] += 1
    d = np.array([x[i]-xn[i] for i in U])
    return d @ M @ d

def run(G, S, sig, T=3000, rng=None):
    n = G.number_of_nodes(); U = [v for v in G if v not in S]
    adj = {v: list(G[v]) for v in G}
    edges = [(i, j) for i, j in G.edges() if not (i in S and j in S)]
    x = rng.random(n); sigU = max(sig[u] for u in U)
    viol, worst, switches, last_switch = 0, 0.0, 0, 0
    topo = None
    for t in range(T):
        xn = hk_step(x, adj, U, sig)
        lhs = V(xn, edges, sigU); rhs = V(x, edges, sigU) - dissip(x, xn, adj, U, sig)
        if lhs > rhs + 1e-10: viol += 1; worst = max(worst, lhs-rhs)
        tp = frozenset((i, j) for i, j in edges if abs(x[i]-x[j]) <= sigU)
        if topo is not None and tp != topo: switches += 1; last_switch = t
        topo = tp; x = xn
    step = np.max(np.abs(hk_step(x, adj, U, sig) - x))
    return viol, worst, switches, last_switch, step

rng = np.random.default_rng(1)
gens = {"ER": lambda: nx.gnp_random_graph(40, 0.12, seed=int(rng.integers(1e9))),
        "BA": lambda: nx.barabasi_albert_graph(40, 2, seed=int(rng.integers(1e9))),
        "tree": lambda: nx.random_labeled_tree(40, seed=int(rng.integers(1e9))),
        "K_n": lambda: nx.complete_graph(25)}
for hetero in [False, True]:
    print("\n=== heterogeneous sigma_U ===" if hetero else "=== common sigma on U ===")
    for name, g in gens.items():
        tot_v, tot_runs, maxsteps, latesw = 0, 0, 0.0, 0
        for r in range(40):
            G = g()
            if not nx.is_connected(G): continue
            S = set(rng.choice(G.number_of_nodes(), size=3, replace=False).tolist())
            s0 = rng.uniform(0.1, 0.4)
            sig = {v: (0.0 if v in S else (rng.uniform(0.05, 0.5) if hetero else s0)) for v in G}
            v, w, sw, ls, st = run(G, S, sig, T=1500, rng=rng)
            tot_runs += 1; tot_v += (v > 0); maxsteps = max(maxsteps, st); latesw = max(latesw, ls)
        print(f"{name:5s} runs={tot_runs:3d}  runs violating lemma={tot_v:3d}  "
              f"max final step={maxsteps:.2e}  latest topology switch t={latesw}")
