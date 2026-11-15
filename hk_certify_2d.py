"""Rigorous freezing certificates for exact d = 2 runs.
Given the final state x(T) and its pattern G, compute the exact fixed point y of T_G
(Dirichlet problems on trapped components, closed-degree weighted means on free ones).
If every edge of E' is strictly on the same side of sigma at y as in G, with margin larger
than 2*rbar, where rbar >= max_i ||x_i(T)-y_i|| (rbar = max l1-distance, rational),
then the ball of radius rbar about y is invariant under T_G and lies in the region of G,
so the interaction graph is frozen from time T on (certificate as in the proof of Thm 5.1).
Reproducible version of the balanced-profile experiment (fixed trial count, fixed seeds), with certificates."""
import itertools, numpy as np, networkx as nx, time
from fractions import Fraction as Fr
from hk_balanced_2d import step, d2, fixes
from hk_critical_edges import solve_exact

def limit_of_pattern(x, adj, S, U, sig2):
    N = {i: [i] + [j for j in adj[i] if d2(x[i], x[j]) <= sig2] for i in U}
    H = nx.Graph(); H.add_nodes_from(U)
    H.add_edges_from((i, j) for i in U for j in N[i] if j in U and j != i)
    y = list(x)
    for comp in nx.connected_components(H):
        comp = sorted(comp); idx = {v: k for k, v in enumerate(comp)}
        if any(j in S for i in comp for j in N[i]):
            for c in (0, 1):
                M = [[Fr(0)]*len(comp) for _ in comp]; b = [Fr(0)]*len(comp)
                for i in comp:
                    M[idx[i]][idx[i]] += len(N[i]) - 1
                    for j in N[i]:
                        if j == i: continue
                        if j in idx: M[idx[i]][idx[j]] -= 1
                        else: b[idx[i]] += x[j][c]
                sol = solve_exact(M, b)
                for v, val in zip(comp, sol):
                    y[v] = (val, y[v][1]) if c == 0 else (y[v][0], val)
        else:
            w = {i: len(N[i]) for i in comp}; W = sum(w.values())
            m = (sum(w[i]*x[i][0] for i in comp)/W, sum(w[i]*x[i][1] for i in comp)/W)
            for i in comp: y[i] = m
    return y

def certificate_data(x, y, Ep, sig):
    """Return rbar and, for every edge, its state and squared margin data (exact strings)."""
    rbar = max(abs(x[i][0]-y[i][0]) + abs(x[i][1]-y[i][1]) for i in range(len(x)))
    edges = [{"edge": [a, b], "active": bool(d2(x[a], x[b]) <= sig**2), "d2_at_y": str(d2(y[a], y[b]))}
             for a, b in Ep]
    return rbar, edges

def certify(x, y, Ep, sig):
    rbar = max(abs(x[i][0]-y[i][0]) + abs(x[i][1]-y[i][1]) for i in range(len(x)))
    for a, b in Ep:
        dx, dy = d2(x[a], x[b]), d2(y[a], y[b])
        if dx <= sig**2:      # active edge: need ||y_ab|| + 2 rbar < sigma
            if not (sig - 2*rbar > 0 and dy < (sig - 2*rbar)**2): return False
        else:                 # inactive edge: need ||y_ab|| - 2 rbar > sigma
            if not dy > (sig + 2*rbar)**2: return False
    return True

if __name__ == "__main__":
    # identical generator and seed to hk_balanced_2d.py
    rng = np.random.default_rng(9); sig = Fr(1); sig2 = Fr(1)
    found = []
    for trial in range(2500000):                   # fixed number of trials: reproducible
        if len(found) >= 48: break
        n = int(rng.integers(4, 8))
        G = nx.gnp_random_graph(n, 0.55, seed=int(rng.integers(1e9)))
        if not nx.is_connected(G): continue
        S = set(rng.choice(n, size=int(rng.integers(1, 5)), replace=False).tolist())
        if len(S) >= n: continue
        U = [v for v in G if v not in S]
        x = [(Fr(int(rng.integers(-5, 6)), 5), Fr(int(rng.integers(-5, 6)), 5)) for _ in range(n)]
        Ep = [(min(a, b), max(a, b)) for a, b in G.edges() if not (a in S and b in S)]
        strict = {e for e in Ep if d2(x[e[0]], x[e[1]]) < sig2}
        crit = [e for e in Ep if d2(x[e[0]], x[e[1]]) == sig2]
        if len(crit) < 2 or len(crit) > 10: continue
        pats = [c for r in range(len(crit)+1) for c in itertools.combinations(crit, r)
                if fixes(G, U, x, strict | set(c))]
        if len(pats) >= 2: found.append((G, S, U, x))
    import json
    records = []
    rngp = np.random.default_rng(10)               # separate stream for perturbations
    runs = cert = 0; last_max = 0; crit_lim = 0
    for (G, S, U, xs) in found[:200]:
        adj = {v: list(G[v]) for v in G}
        Ep = [(min(a, b), max(a, b)) for a, b in G.edges() if not (a in S and b in S)]
        for rep in range(3):
            x = list(xs)
            for i in U:
                x[i] = (x[i][0] + Fr(int(rngp.integers(-40, 41)), 10000),
                        x[i][1] + Fr(int(rngp.integers(-40, 41)), 10000))
            pat = lambda z: frozenset(e for e in Ep if d2(z[e[0]], z[e[1]]) <= sig2)
            p, last = pat(x), 0
            for t in range(1, 301):
                x = step(x, adj, U); q = pat(x)
                if q != p: last = t
                p = q
                if t - last > 100: break
            y = limit_of_pattern(x, adj, S, U, sig2)
            runs += 1; last_max = max(last_max, last); c = certify(x, y, Ep, sig); cert += c
            crit_lim += any(d2(y[a], y[b]) == sig2 for a, b in Ep)
            rbar, edata = certificate_data(x, y, Ep, sig)
            records.append({"profile": len(records) // 3, "graph_edges": sorted(G.edges()),
                            "stubborn": sorted(S), "T": t, "last_change": last,
                            "x_T": [[str(c) for c in v] for v in x], "y": [[str(c) for c in v] for v in y],
                            "rbar": str(rbar), "edges": edata, "certified": bool(c),
                            "candidate_limit_has_critical_edge": any(d2(y[a], y[b]) == sig2 for a, b in Ep)})
    with open("hk_certificates_2d.json", "w") as f: json.dump(records, f)
    print(f"profiles: {len(found)} | runs: {runs} | latest change: {last_max} | certified frozen: {cert}"
          f" | limits with critical edges: {crit_lim}")
