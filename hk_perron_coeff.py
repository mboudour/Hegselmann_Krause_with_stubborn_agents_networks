"""For each critical edge (limit gap exactly sigma, gap not identically zero):
under the frozen topology, e(t)=x_U(t)-y_U obeys e(t+1)=A e(t), A=D^{-1}M with M
symmetric, so g(t)=s*(e_i-e_j)(t)=sum_k a_k lambda_k^t (real spectrum).
We report which eigenvalue dominates g, whether it is the Perron root of the
relevant component, and whether a +/- lambda pair (possible sign alternation) occurs."""
import numpy as np, networkx as nx
from fractions import Fraction as Fr
from hk_critical_edges import nbrs, step, topo, frozen_limit

rng = np.random.default_rng(11); sig = Fr(1, 4); q = 24; T = 120
out = dict(edges=0, perron_dominant=0, subdominant=0, pm_pair=0, neg_dominant=0)
examples = []
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
    tf = max([t for t in range(1, T+1) if tps[t] != tps[t-1]], default=0)
    y, N = frozen_limit(traj[-1], G, S, U, sig)
    crit = [(i, j) for i, j in edges if abs(y[i]-y[j]) == sig
            and any(abs(z[i]-z[j]) != sig for z in traj[tf:])]
    if not crit: continue
    idx = {v: k for k, v in enumerate(U)}; m = len(U)
    d = np.array([len(N[i]) for i in U], float)
    M = np.zeros((m, m))
    for i in U:
        for j in N[i]:
            if j in idx: M[idx[i], idx[j]] = 1
    Dm = np.diag(d**-0.5); Bsym = Dm @ M @ Dm          # A = D^{-1/2} Bsym D^{1/2}
    lam, Q = np.linalg.eigh(Bsym)
    e0 = np.array([float(traj[tf][i] - y[i]) for i in U])
    z0 = Q.T @ (np.sqrt(d) * e0)                       # modal coordinates
    for i, j in crit:
        s = np.sign(float(y[i]-y[j]))
        c = np.zeros(m)
        if i in idx: c[idx[i]] += s
        if j in idx: c[idx[j]] -= s
        w = Q.T @ (Dm @ c)                              # g(t) = sum_k w_k z0_k lam_k^t
        a = w * z0
        nz = np.abs(a) > 1e-12 * max(1e-300, np.abs(a).max())
        if not nz.any(): continue
        out["edges"] += 1
        mod = np.abs(lam[nz]).max()
        dom = lam[nz][np.isclose(np.abs(lam[nz]), mod, atol=1e-9)]
        # Perron root of the component(s) touching the edge
        H = nx.Graph(); H.add_nodes_from(U)
        H.add_edges_from((p, r) for p in U for r in N[p] if r in U and r != p)
        comp = set().union(*[nx.node_connected_component(H, v) for v in (i, j) if v in idx])
        ci = [idx[v] for v in comp]
        rho = max(abs(np.linalg.eigvalsh(Bsym[np.ix_(ci, ci)])))
        if np.isclose(dom.max(), rho, atol=1e-9) and len(dom) == 1: out["perron_dominant"] += 1
        else:
            out["subdominant"] += 1
            examples.append((G.edges(), sorted(S), (i, j), dom, rho))
        if len(dom) == 2: out["pm_pair"] += 1
        if dom.max() < 0: out["neg_dominant"] += 1
print(out)
for ex in examples[:5]:
    print("degenerate example: edges", list(ex[0]), "S", ex[1], "crit edge", ex[2],
          "dominant lambdas", np.round(ex[3], 4), "component Perron", round(ex[4], 4))
