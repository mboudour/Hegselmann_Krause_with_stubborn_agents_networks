"""Exact-arithmetic counterexample search for per-edge monotonicity
(an edge that switches on never switches off) on larger graphs (n=9..15).
Initial conditions designed to provoke break-ups: opinions on near-sigma chains,
planted ties, and stubborn agents at both extremes pulling clusters apart."""
import numpy as np, networkx as nx, time
from fractions import Fraction as Fr
from hk_critical_edges import step

rng = np.random.default_rng(101); T = 60; q = 48
def onset(x, edges, sig): return {e: abs(x[e[0]]-x[e[1]]) <= sig for e in edges}

tot = dict(runs=0, on=0, off=0, on_then_off_edges=0)
latest_off, latest_on, bad = 0, 0, []
t0 = time.time()
while time.time() - t0 < 480:
    n = int(rng.integers(9, 16)); kind = int(rng.integers(0, 5))
    G = [nx.path_graph(n), nx.cycle_graph(n), nx.random_labeled_tree(n, seed=int(rng.integers(1e9))),
         nx.gnp_random_graph(n, 0.3, seed=int(rng.integers(1e9))),
         nx.watts_strogatz_graph(n, 4, 0.2, seed=int(rng.integers(1e9)))][kind]
    if not nx.is_connected(G): continue
    sig = Fr(int(rng.integers(3, 13)), 48)
    k = int(rng.integers(1, 4)); S = set(rng.choice(n, size=k, replace=False).tolist())
    U = [v for v in G if v not in S]; adj = {v: list(G[v]) for v in G}
    edges = [e for e in G.edges() if not (e[0] in S and e[1] in S)]
    mode = int(rng.integers(0, 3))
    if mode == 0:     # BFS chain with steps just below / at sigma
        order = list(nx.bfs_tree(G, 0)); x = [Fr(0)]*n
        for v in order[1:]:
            p = next(iter(nx.bfs_tree(G, 0).predecessors(v)))
            x[v] = x[p] + sig*Fr(int(rng.integers(20, 25)), 24)*(1 if rng.random() < .5 else -1)
    elif mode == 1:   # random grid + planted ties
        x = [Fr(int(rng.integers(0, q+1)), q) for _ in range(n)]
        for (i, j) in list(G.edges())[: n]:
            if rng.random() < .6: x[j] = x[i] + (sig if rng.random() < .5 else -sig)
    else:             # stubborn agents at both extremes
        x = [Fr(int(rng.integers(q//4, 3*q//4+1)), q) for _ in range(n)]
        for idx, s in enumerate(sorted(S)): x[s] = Fr(0) if idx % 2 == 0 else Fr(1)
    st = onset(x, edges, sig); hist = {e: [] for e in edges}
    for t in range(1, T+1):
        x = step(x, adj, U, sig)
        # cap denominators by exact rounding? no: keep exact to avoid artefacts
        new = onset(x, edges, sig)
        for e in edges:
            if new[e] != st[e]: hist[e].append((t, new[e]))
        st = new
    tot["runs"] += 1
    for e in edges:
        for t, on in hist[e]:
            if on: tot["on"] += 1; latest_on = max(latest_on, t)
            else: tot["off"] += 1; latest_off = max(latest_off, t)
        seq = [on for _, on in hist[e]]
        if True in seq and False in seq[seq.index(True):]:
            tot["on_then_off_edges"] += 1
            bad.append((n, kind, mode, str(sig), sorted(S), e, hist[e]))
print(tot, "| latest off:", latest_off, "| latest on:", latest_on)
for b in bad[:5]: print("ON->OFF:", b)
