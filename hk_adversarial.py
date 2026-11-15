"""Adversarial search for persistent topology switching in network HK with
stubborn agents (common bound sigma on U, sigma=0 on S).
Stage 1: float random search with ties planted at distance exactly sigma.
Stage 2: exact rational re-run of the worst cases (no rounding artefacts)."""
import numpy as np, networkx as nx
from fractions import Fraction as Fr

def step(x, adj, U, sig):
    xn = list(x)
    for i in U:
        nb = [j for j in adj[i] if abs(x[i]-x[j]) <= sig] + [i]
        xn[i] = sum(x[j] for j in nb) / len(nb)
    return xn

def topo(x, edges, sig):
    return frozenset(e for e in edges if abs(x[e[0]]-x[e[1]]) <= sig)

def simulate(G, S, x0, sig, T):
    adj = {v: list(G[v]) for v in G}; U = [v for v in G if v not in S]
    edges = [e for e in G.edges() if not (e[0] in S and e[1] in S)]
    x, tp, sw = list(x0), None, []
    seen = {}
    for t in range(T):
        new = topo(x, edges, sig)
        if tp is not None and new != tp: sw.append(t)
        tp = new
        key = tuple(x)
        if key in seen:                       # exact periodic orbit
            return sw, ("cycle" if t - seen[key] > 1 else "fixed", t - seen[key])
        seen[key] = t
        x = step(x, adj, U, sig)
    return sw, ("none", 0)

def planted_x0(G, S, sig, rng, exact=False):
    n = G.number_of_nodes(); q = 24
    x = [Fr(int(rng.integers(0, q+1)), q) for _ in range(n)]   # rational grid
    for s in S: x[s] = Fr(int(rng.integers(0, q+1)), q)
    # plant ties: for a few edges set |x_i - x_j| = sigma exactly
    for (i, j) in list(G.edges())[: max(1, n // 3)]:
        if j not in S and rng.random() < 0.7:
            x[j] = x[i] + (sig if rng.random() < .5 else -sig)
    return x if exact else [float(v) for v in x]

rng = np.random.default_rng(7)
sig = Fr(1, 4)
families = []
for n in range(4, 9):
    families += [("path", nx.path_graph(n)), ("star", nx.star_graph(n-1)),
                 ("cycle", nx.cycle_graph(n))]
    for _ in range(3):
        families.append(("tree", nx.random_labeled_tree(n, seed=int(rng.integers(1e9)))))

records = []
for name, G in families:
    n = G.number_of_nodes()
    for trial in range(400):
        S = set(rng.choice(n, size=2, replace=False).tolist())
        x0 = planted_x0(G, S, sig, rng, exact=True)
        sw, cyc = simulate(G, S, [float(v) for v in x0], float(sig), T=600)
        last = sw[-1] if sw else 0
        records.append((cyc[0] == "cycle", last, len(sw), name, n, G, S, x0))

records.sort(key=lambda r: (r[0], r[1]), reverse=True)
print("Stage 1 (float): cycles found:", sum(r[0] for r in records),
      "| total runs:", len(records))
print("worst latest-switch times:", [r[1] for r in records[:10]])

print("\nStage 2 (exact rationals) on 10 worst cases:")
for cyc_f, last, nsw, name, n, G, S, x0 in records[:10]:
    sw, cyc = simulate(G, S, x0, sig, T=400)
    print(f"{name:5s} n={n} S={sorted(S)} float_last={last:4d} "
          f"exact_switches={len(sw):3d} exact_last={sw[-1] if sw else 0:4d} cycle={cyc}")
