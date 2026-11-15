"""Exact-arithmetic test of the one-sided approach conjecture for network HK
with stubborn agents (sigma=0 on S, common sigma on U).
For runs whose limit y has critical edges (|y_i - y_j| = sigma exactly) we record
whether the gap g(t)=|x_i-x_j|-sigma keeps a constant sign after the topology
freezes, and compare its decay rate with the Perron root of the frozen A_UU."""
import numpy as np, networkx as nx
from fractions import Fraction as Fr

def nbrs(x, adj, i, sig):
    return [j for j in adj[i] if abs(x[i]-x[j]) <= sig] + [i]

def step(x, adj, U, sig):
    xn = list(x)
    for i in U:
        nb = nbrs(x, adj, i, sig); xn[i] = sum(x[j] for j in nb) / len(nb)
    return xn

def topo(x, edges, sig):
    return frozenset(e for e in edges if abs(x[e[0]]-x[e[1]]) <= sig)

def solve_exact(M, b):                       # Gaussian elimination over Q
    n = len(b); M = [row[:] + [b[k]] for k, row in enumerate(M)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0); M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c] / M[c][c]; M[r] = [a - f*bb for a, bb in zip(M[r], M[c])]
    return [M[k][n] / M[k][k] for k in range(n)]

def frozen_limit(x, G, S, U, sig):
    adj = {v: list(G[v]) for v in G}
    N = {i: nbrs(x, adj, i, sig) for i in U}
    H = nx.Graph(); H.add_nodes_from(U)
    H.add_edges_from((i, j) for i in U for j in N[i] if j in U and j != i)
    y = list(x)
    for comp in nx.connected_components(H):
        comp = sorted(comp)
        if any(j in S for i in comp for j in N[i]):        # Dirichlet problem
            idx = {v: k for k, v in enumerate(comp)}
            M = [[Fr(0)]*len(comp) for _ in comp]; b = [Fr(0)]*len(comp)
            for i in comp:
                M[idx[i]][idx[i]] += len(N[i]) - 1
                for j in N[i]:
                    if j == i: continue
                    if j in idx: M[idx[i]][idx[j]] -= 1
                    else: b[idx[i]] += x[j]
            for v, val in zip(comp, solve_exact(M, b)): y[v] = val
        else:                                                # free cluster: degree-weighted mean
            w = {i: len(N[i]) for i in comp}
            m = sum(w[i]*x[i] for i in comp) / sum(w.values())
            for i in comp: y[i] = m
    return y, N

def perron(N, U):
    idx = {v: k for k, v in enumerate(U)}; A = np.zeros((len(U), len(U)))
    for i in U:
        for j in N[i]:
            if j in idx: A[idx[i], idx[j]] += 1/len(N[i])
    ev = np.linalg.eigvals(A); return max(abs(ev)), min(ev.real)

if __name__ == "__main__":
    rng = np.random.default_rng(11); sig = Fr(1, 4); q = 24
    T = 120
    stats = dict(runs=0, with_crit=0, crit_edges=0, ident_zero=0,
                 one_sided=0, sign_flip_after_freeze=0, limit_mismatch=0)
    rates = []
    for trial in range(1500):
        n = int(rng.integers(4, 8))
        G = [nx.path_graph(n), nx.cycle_graph(n), nx.star_graph(n-1),
             nx.random_labeled_tree(n, seed=int(rng.integers(1e9)))][trial % 4]
        S = set(rng.choice(n, size=2, replace=False).tolist()); U = [v for v in G if v not in S]
        adj = {v: list(G[v]) for v in G}
        edges = [e for e in G.edges() if not (e[0] in S and e[1] in S)]
        x = [Fr(int(rng.integers(0, q+1)), q) for _ in range(n)]
        for (i, j) in list(G.edges())[: max(1, n//2)]:
            if j not in S: x[j] = x[i] + (sig if rng.random() < .5 else -sig)
        traj, tps = [x], [topo(x, edges, sig)]
        for t in range(T):
            x = step(x, adj, U, sig); traj.append(x); tps.append(topo(x, edges, sig))
        t_freeze = max([t for t in range(1, T+1) if tps[t] != tps[t-1]], default=0)
        y, N = frozen_limit(traj[-1], G, S, U, sig)
        stats["runs"] += 1
        # sanity: frozen-topology limit must be admissible for that topology
        if topo(y, edges, sig) != tps[-1] and not any(abs(abs(y[i]-y[j]) - sig) == 0 for i, j in edges):
            stats["limit_mismatch"] += 1
        crit = [(i, j) for i, j in edges if abs(y[i]-y[j]) == sig]
        if not crit: continue
        stats["with_crit"] += 1; stats["crit_edges"] += len(crit)
        rho, lmin = perron(N, U)
        for i, j in crit:
            g = [abs(z[i]-z[j]) - sig for z in traj[t_freeze:]]
            if all(v == 0 for v in g): stats["ident_zero"] += 1; continue
            signs = {np.sign(float(v)) for v in g if v != 0}
            if len(signs) == 1: stats["one_sided"] += 1
            else: stats["sign_flip_after_freeze"] += 1
            nz = [abs(float(v)) for v in g if v != 0]
            if len(nz) > 20: rates.append((nz[-1]/nz[-11])**0.1 / rho if rho > 0 else np.nan)

    print(stats)
    if rates:
        r = np.array([v for v in rates if np.isfinite(v)])
        print(f"observed gap decay rate / Perron root of frozen A_UU: "
              f"median={np.median(r):.4f}, min={r.min():.4f}, max={r.max():.4f}, n={len(r)}")
