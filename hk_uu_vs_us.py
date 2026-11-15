"""Split the on->off test by edge type and measure the Lyapunov drop at releases.
(i)  U-U edges: does an edge that switched on ever switch off?
(ii) U-S edges: at each release (on->off) record the energy drop
     V(x(t)) - V(x(t+1)) and the dissipation delta^T (D_U + A_UU) delta,
     to see whether releases cost a uniformly positive amount of energy."""
import numpy as np, networkx as nx, time
from fractions import Fraction as Fr
from hk_critical_edges import step

def onset(x, edges, sig): return {e: abs(x[e[0]]-x[e[1]]) <= sig for e in edges}
def V(x, edges, sig): return sum(min((x[i]-x[j])**2, sig**2) for i, j in edges)

rng = np.random.default_rng(2027); T = 60; q = 48
tot = dict(runs=0, uu_on=0, uu_off=0, us_on=0, us_off=0, uu_on_then_off=0, us_on_then_off=0)
rel_drop, uu_bad = [], []
for attempt in range(55000):                  # fixed number of attempts: reproducible
    n = int(rng.integers(9, 16)); kind = int(rng.integers(0, 5))
    G = [nx.path_graph(n), nx.cycle_graph(n), nx.random_labeled_tree(n, seed=int(rng.integers(1e9))),
         nx.gnp_random_graph(n, 0.3, seed=int(rng.integers(1e9))),
         nx.watts_strogatz_graph(n, 4, 0.2, seed=int(rng.integers(1e9)))][kind]
    if not nx.is_connected(G): continue
    sig = Fr(int(rng.integers(3, 13)), 48)
    S = set(rng.choice(n, size=int(rng.integers(1, 4)), replace=False).tolist())
    U = [v for v in G if v not in S]; adj = {v: list(G[v]) for v in G}
    edges = [e for e in G.edges() if not (e[0] in S and e[1] in S)]
    mode = int(rng.integers(0, 3))
    if mode == 0:
        B = nx.bfs_tree(G, 0); x = [Fr(0)]*n
        for v in list(B)[1:]:
            p = next(iter(B.predecessors(v)))
            x[v] = x[p] + sig*Fr(int(rng.integers(20, 25)), 24)*(1 if rng.random() < .5 else -1)
    elif mode == 1:
        x = [Fr(int(rng.integers(0, q+1)), q) for _ in range(n)]
        for (i, j) in list(G.edges())[: n]:
            if rng.random() < .6: x[j] = x[i] + (sig if rng.random() < .5 else -sig)
    else:
        x = [Fr(int(rng.integers(q//4, 3*q//4+1)), q) for _ in range(n)]
        for k, s in enumerate(sorted(S)): x[s] = Fr(0) if k % 2 == 0 else Fr(1)
    st = onset(x, edges, sig); was_on = {e: False for e in edges}
    for t in range(1, T+1):
        xn = step(x, adj, U, sig); new = onset(xn, edges, sig)
        for e in edges:
            us = e[0] in S or e[1] in S
            if new[e] != st[e]:
                key = ("us_" if us else "uu_") + ("on" if new[e] else "off")
                tot[key] += 1
                if new[e]: was_on[e] = True
                elif was_on[e]:
                    tot["us_on_then_off" if us else "uu_on_then_off"] += 1
                    if us:
                        rel_drop.append(float((V(x, edges, sig) - V(xn, edges, sig)) / sig**2))
                    else:
                        uu_bad.append((n, kind, mode, str(sig), sorted(S), e, t))
        x, st = xn, new
    tot["runs"] += 1
print(tot)
if rel_drop:
    r = np.array(rel_drop)
    print(f"U-S releases: energy drop at release step / sigma^2: min={r.min():.3e} "
          f"median={np.median(r):.3e} max={r.max():.3e} (n={len(r)})")
for b in uu_bad[:5]: print("U-U on->off:", b)
