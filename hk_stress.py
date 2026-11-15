"""Stress test: try to force sign alternation at a critical edge.
A free cluster C (all in U) sits at consensus m; a stubborn node s is attached to a
node a of C with x_s = m + sigma (critical distance). We perturb the cluster along
the eigenvector of the most negative eigenvalue of its frozen A_C, so the gap
|x_a - x_s| - sigma alternates in sign if the frozen dynamics persisted.
Exact rational arithmetic; we count topology switches and check convergence."""
import numpy as np, networkx as nx
from fractions import Fraction as Fr
from hk_critical_edges import step, topo

sig = Fr(1, 4); T = 200
def lazy_rw(Gc):
    nodes = list(Gc); k = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((len(nodes),)*2)
    for v in nodes:
        nb = list(Gc[v]) + [v]
        for w in nb: A[k[v], k[w]] = 1/len(nb)
    return nodes, A

results = []
cands = [("star3", nx.star_graph(3)), ("star5", nx.star_graph(5)), ("path3", nx.path_graph(3)),
         ("path4", nx.path_graph(4)), ("K22", nx.complete_bipartite_graph(2, 2)),
         ("K23", nx.complete_bipartite_graph(2, 3)), ("cycle4", nx.cycle_graph(4))]
for name, C in cands:
    nodes, A = lazy_rw(C)
    lam, V = np.linalg.eig(A); lam = lam.real; V = V.real
    kneg = int(np.argmin(lam))
    if lam[kneg] >= 0: results.append((name, "no negative eigenvalue")); continue
    v = V[:, kneg] / np.abs(V[:, kneg]).max()
    for a in nodes:                                   # attach stubborn to each node
        for eps_den in (64, 256, 1024):
            for sgn in (1, -1):
                G = C.copy(); s = max(nodes) + 1; G.add_edge(a, s)
                U = nodes; adj = {u: list(G[u]) for u in G}
                edges = [e for e in G.edges()]
                m = Fr(1, 2)
                x = {u: m + sgn*Fr(round(v[i]*1000), 1000)/eps_den for i, u in enumerate(nodes)}
                x[s] = m + sig
                xs = [x[u] for u in range(s+1)]
                # require the critical edge to start "off" (gap > sigma), else skip
                if abs(xs[a]-xs[s]) <= sig: continue
                tps = [topo(xs, edges, sig)]; traj = [xs]
                for t in range(T):
                    xs = step(xs, adj, U, sig); traj.append(xs); tps.append(topo(xs, edges, sig))
                sw = [t for t in range(1, T+1) if tps[t] != tps[t-1]]
                fin = float(max(abs(traj[-1][u]-traj[-2][u]) for u in U))
                results.append((name, a, eps_den, sgn, round(float(lam[kneg]), 4),
                                len(sw), sw[-1] if sw else 0, fin))
for r in results: print(r)
mx = max((r[5] for r in results if len(r) > 2), default=0)
print("\nmax number of exact topology switches over all constructions:", mx)
