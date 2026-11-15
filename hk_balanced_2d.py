"""Exploratory search (time-budgeted; the reproducible version is hk_certify_2d.py).
Conjecture 5.10 (d = 2): exact rational search at balanced configurations.
Opinions on the lattice (1/5)Z^2 with sigma = 1, so that critical directions include the
Pythagorean vectors (+-1,0),(0,+-1),(+-3/5,+-4/5),(+-4/5,+-3/5) and signed zero-sum relations
among critical neighbours are common. We keep profiles fixed by >= 2 patterns that differ on
critical edges (balanced: the only limits allowing persistent switching, Theorem 5.3), perturb
the regular agents by small rationals and run exact HK, recording late switching."""
import itertools, numpy as np, networkx as nx, time
from fractions import Fraction as Fr

rng = np.random.default_rng(9); sig2 = Fr(1)
def d2(a, b): return (a[0]-b[0])**2 + (a[1]-b[1])**2
def fixes(G, U, x, Gam):
    for i in U:
        nb = [i] + [j for j in G[i] if (min(i, j), max(i, j)) in Gam]
        if sum(x[j][0]-x[i][0] for j in nb) != 0 or sum(x[j][1]-x[i][1] for j in nb) != 0:
            return False
    return True
def step(x, adj, U):
    y = list(x)
    for i in U:
        nb = [i] + [j for j in adj[i] if d2(x[i], x[j]) <= sig2]
        y[i] = (sum(x[j][0] for j in nb)/len(nb), sum(x[j][1] for j in nb)/len(nb))
    return y

if __name__ == "__main__":
    found, t0 = [], time.time()
    while time.time() - t0 < 200:
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
    print("balanced 2-d configurations found:", len(found))

    T = 300; runs = late = 0; maxlast = 0
    for (G, S, U, xs) in found[:200]:
        adj = {v: list(G[v]) for v in G}
        Ep = [(min(a, b), max(a, b)) for a, b in G.edges() if not (a in S and b in S)]
        for rep in range(3):
            x = list(xs)
            for i in U:
                x[i] = (x[i][0] + Fr(int(rng.integers(-40, 41)), 10000),
                        x[i][1] + Fr(int(rng.integers(-40, 41)), 10000))
            pat = lambda z: frozenset(e for e in Ep if d2(z[e[0]], z[e[1]]) <= sig2)
            p, last = pat(x), 0
            for t in range(1, T+1):
                x = step(x, adj, U); q = pat(x)
                if q != p: last = t
                p = q
                if t - last > 100: break
            runs += 1; maxlast = max(maxlast, last); late += last > T - 50
    print(f"exact 2-d runs: {runs} | still switching near horizon: {late} | latest switch: {maxlast}")
