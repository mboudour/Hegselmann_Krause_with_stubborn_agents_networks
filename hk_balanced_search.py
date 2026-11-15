"""Search for failure of finite-time freezing (Conjecture 5.6), d = 1.
Step 1: random small graphs with opinions on the lattice (sigma/2)Z; keep profiles x*
        that are fixed by at least two patterns differing on critical edges
        (balanced configurations, the only ones allowed to switch by Theorem 5.2).
Step 2: run exact HK from x* + small rational perturbations of U and record
        whether the interaction graph is still switching near the horizon."""
import itertools, numpy as np, networkx as nx
from fractions import Fraction as Fr
from hk_critical_edges import step

sig = Fr(1); rng = np.random.default_rng(5)
def fixes(G, S, U, x, Gam):
    for i in U:
        nb = [i] + [j for j in G[i] if (min(i, j), max(i, j)) in Gam]
        if sum(x[j] - x[i] for j in nb) != 0: return False
    return True

found = []
for trial in range(20000):
    n = int(rng.integers(4, 8))
    G = nx.gnp_random_graph(n, 0.5, seed=int(rng.integers(1e9)))
    if not nx.is_connected(G): continue
    S = set(rng.choice(n, size=int(rng.integers(1, 4)), replace=False).tolist())
    U = [v for v in G if v not in S]
    x = [Fr(int(rng.integers(0, 7)), 2) for _ in range(n)]
    Ep = [(min(a, b), max(a, b)) for a, b in G.edges() if not (a in S and b in S)]
    strict = {e for e in Ep if abs(x[e[0]] - x[e[1]]) < sig}
    crit = [e for e in Ep if abs(x[e[0]] - x[e[1]]) == sig]
    if len(crit) < 2 or len(crit) > 10: continue
    pats = [frozenset(strict | set(c)) for r in range(len(crit) + 1)
            for c in itertools.combinations(crit, r) if fixes(G, S, U, x, strict | set(c))]
    if len(pats) >= 2: found.append((G, S, U, x, pats, crit))
print("balanced configurations found:", len(found))

T = 400; late = []
for (G, S, U, xs, pats, crit) in found[:300]:
    adj = {v: list(G[v]) for v in G}
    Ep = [(min(a, b), max(a, b)) for a, b in G.edges() if not (a in S and b in S)]
    for rep in range(4):
        x = list(xs)
        for i in U: x[i] += Fr(int(rng.integers(-50, 51)), 10000)
        def pat(z): return frozenset(e for e in Ep if abs(z[e[0]] - z[e[1]]) <= sig)
        p = pat(x); sw = []
        for t in range(1, T + 1):
            x = step(x, adj, U, sig); q = pat(x)
            if q != p: sw.append(t)
            p = q
            if t > 150 and (not sw or sw[-1] < t - 100): break   # frozen
        if sw and sw[-1] > T - 50:
            late.append((sorted(G.edges()), sorted(S), [str(v) for v in xs], len(sw), sw[-5:]))
print("perturbed runs still switching near the horizon:", len(late))
for l in late[:5]: print(l)
