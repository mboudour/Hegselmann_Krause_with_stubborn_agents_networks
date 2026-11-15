"""Exact check (d = 2) of the local energy identity of Lemma 5.8:
for a profile x* fixed by every pattern in a set P and a deviation e with Gamma(x*+e) in P,
  V(x*+e) - V(x*) = Q_{Gamma(x*+e)}(e) = min_{G in P} Q_G(e),
and V(x*+e) - V(x*) <= Q_G(e) for every G in P.
Profiles: balanced lattice configurations from hk_balanced_2d (fixed by >= 2 patterns)."""
import itertools, numpy as np, networkx as nx
from fractions import Fraction as Fr
from hk_balanced_2d import d2, fixes

rng = np.random.default_rng(21); sig2 = Fr(1)
def V(x, Ep): return sum(min(d2(x[a], x[b]), sig2) for a, b in Ep)
def Q(e, G): return sum(d2(e[a], e[b]) for a, b in G)
checked = eq_ok = ineq_ok = 0; cfg = 0
while cfg < 60:
    n = int(rng.integers(4, 8)); G = nx.gnp_random_graph(n, 0.55, seed=int(rng.integers(1e9)))
    if not nx.is_connected(G): continue
    S = set(rng.choice(n, size=int(rng.integers(1, 5)), replace=False).tolist())
    if len(S) >= n: continue
    U = [v for v in G if v not in S]
    x = [(Fr(int(rng.integers(-5, 6)), 5), Fr(int(rng.integers(-5, 6)), 5)) for _ in range(n)]
    Ep = [(min(a, b), max(a, b)) for a, b in G.edges() if not (a in S and b in S)]
    strict = {e for e in Ep if d2(x[e[0]], x[e[1]]) < sig2}
    crit = [e for e in Ep if d2(x[e[0]], x[e[1]]) == sig2]
    if not crit or len(crit) > 10: continue
    P = [frozenset(strict | set(c)) for r in range(len(crit)+1)
         for c in itertools.combinations(crit, r) if fixes(G, U, x, strict | set(c))]
    if len(P) < 2: continue
    cfg += 1
    for _ in range(400):
        e = [(Fr(0), Fr(0)) if v in S else (Fr(int(rng.integers(-60, 61)), 10000),
             Fr(int(rng.integers(-60, 61)), 10000)) for v in range(n)]
        xe = [(x[v][0]+e[v][0], x[v][1]+e[v][1]) for v in range(n)]
        Gam = frozenset(f for f in Ep if d2(xe[f[0]], xe[f[1]]) <= sig2)
        dV = V(xe, Ep) - V(x, Ep)
        ineq_ok += all(dV <= Q(e, Gp) for Gp in P)
        if Gam in P:
            checked += 1; eq_ok += (dV == Q(e, Gam) == min(Q(e, Gp) for Gp in P))
print(f"configurations: {cfg} | deviations with Gamma in P: {checked}, identity exact in {eq_ok}"
      f" | upper bound V-V* <= Q_G for all G in P held in {ineq_ok} of {cfg*400}")
