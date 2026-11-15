"""Follow-up on on->off counterexamples: rerun the generator, keep the violating
runs, classify the violating edge (U-U or U-S), and continue them exactly to
T=300 to see whether switching stops (finite) or keeps going."""
import numpy as np, networkx as nx, time
from fractions import Fraction as Fr
from hk_critical_edges import step
# same generator and seed as hk_capture_large.py
rng = np.random.default_rng(101); q = 48
def onset(x, edges, sig): return {e: abs(x[e[0]]-x[e[1]]) <= sig for e in edges}
found, t0 = [], time.time()
while len(found) < 17 and time.time() - t0 < 300:
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
    if mode == 0:
        order = list(nx.bfs_tree(G, 0)); x = [Fr(0)]*n
        for v in order[1:]:
            p = next(iter(nx.bfs_tree(G, 0).predecessors(v)))
            x[v] = x[p] + sig*Fr(int(rng.integers(20, 25)), 24)*(1 if rng.random() < .5 else -1)
    elif mode == 1:
        x = [Fr(int(rng.integers(0, q+1)), q) for _ in range(n)]
        for (i, j) in list(G.edges())[: n]:
            if rng.random() < .6: x[j] = x[i] + (sig if rng.random() < .5 else -sig)
    else:
        x = [Fr(int(rng.integers(q//4, 3*q//4+1)), q) for _ in range(n)]
        for idx, s in enumerate(sorted(S)): x[s] = Fr(0) if idx % 2 == 0 else Fr(1)
    x0 = list(x); st = onset(x, edges, sig); hist = {e: [] for e in edges}
    for t in range(1, 61):
        x = step(x, adj, U, sig); new = onset(x, edges, sig)
        for e in edges:
            if new[e] != st[e]: hist[e].append((t, new[e]))
        st = new
    viol = [e for e in edges if True in [o for _, o in hist[e]] and
            False in [o for _, o in hist[e]][[o for _, o in hist[e]].index(True):]]
    if viol: found.append((G, S, U, adj, edges, sig, x0, viol))

TT = 300
for G, S, U, adj, edges, sig, x0, viol in found:
    x = list(x0); st = onset(x, edges, sig); sw = []
    for t in range(1, TT+1):
        x = step(x, adj, U, sig); new = onset(x, edges, sig)
        sw += [t for e in edges if new[e] != st[e]]; st = new
    types = ["U-S" if (e[0] in S or e[1] in S) else "U-U" for e in viol]
    late = [t for t in sw if t > 60]
    print(f"n={G.number_of_nodes():2d} |S|={len(S)} sigma={sig} violating={types} "
          f"total switches={len(sw):3d} last switch t={max(sw)} switches after t=60: {len(late)}")
