"""Exact-arithmetic test of the capture lemma for network HK with stubborn agents:
after some finite T0, edges only switch on (never off).
Records every switch with direction and time; flags edges that switch on and later off."""
import numpy as np, networkx as nx
from fractions import Fraction as Fr
from hk_critical_edges import step

rng = np.random.default_rng(23); q = 24; T = 150
def onset(x, edges, sig): return {e: abs(x[e[0]]-x[e[1]]) <= sig for e in edges}

tot = dict(runs=0, on=0, off=0, on_then_off_edges=0, runs_off_after_on=0,
           runs_off_after_t10=0)
last_off, last_on, bad = [], [], []
for trial in range(1200):
    n = int(rng.integers(4, 9)); kind = trial % 5
    if kind == 4:
        G = nx.gnp_random_graph(n, 0.4, seed=int(rng.integers(1e9)))
        if not nx.is_connected(G): continue
    else:
        G = [nx.path_graph(n), nx.cycle_graph(n), nx.star_graph(n-1),
             nx.random_labeled_tree(n, seed=int(rng.integers(1e9)))][kind]
    sig = Fr(int(rng.integers(2, 9)), 24)
    S = set(rng.choice(n, size=int(rng.integers(1, 3)), replace=False).tolist())
    U = [v for v in G if v not in S]; adj = {v: list(G[v]) for v in G}
    edges = [e for e in G.edges() if not (e[0] in S and e[1] in S)]
    x = [Fr(int(rng.integers(0, q+1)), q) for _ in range(n)]
    for (i, j) in list(G.edges())[: max(1, n//2)]:
        if j not in S and rng.random() < .6: x[j] = x[i] + (sig if rng.random() < .5 else -sig)
    st = onset(x, edges, sig); hist = {e: [] for e in edges}
    for t in range(1, T+1):
        x = step(x, adj, U, sig); new = onset(x, edges, sig)
        for e in edges:
            if new[e] != st[e]: hist[e].append((t, "on" if new[e] else "off"))
        st = new
    tot["runs"] += 1
    evs = [(t, d, e) for e in edges for t, d in hist[e]]
    ons = [t for t, d, _ in evs if d == "on"]; offs = [t for t, d, _ in evs if d == "off"]
    tot["on"] += len(ons); tot["off"] += len(offs)
    if offs: last_off.append(max(offs))
    if ons: last_on.append(max(ons))
    if offs and ons and max(offs) > min(ons): tot["runs_off_after_on"] += 1
    if offs and max(offs) > 10: tot["runs_off_after_t10"] += 1
    for e in edges:
        seq = [d for _, d in hist[e]]
        if "on" in seq and "off" in seq[seq.index("on"):]:
            tot["on_then_off_edges"] += 1
            bad.append((list(G.edges()), sorted(S), str(sig), e, hist[e]))
print(tot)
print("latest off-switch over all runs:", max(last_off, default=0),
      "| latest on-switch over all runs:", max(last_on, default=0))
print("on-switch times > 10:", sorted(t for t in last_on if t > 10)[-10:])
for b in bad[:4]: print("on->off edge example:", b)
