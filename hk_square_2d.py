"""Targeted 2-d test: a regular agent c at the origin, joined to four stubborn agents at
(+-1,0),(0,+-1) (critical, balanced in two ways) and to a clique of m regular agents near
the origin. Small rational perturbations; exact arithmetic; we record pattern switches."""
import numpy as np
from fractions import Fraction as Fr
from hk_balanced_2d import step, d2
rng = np.random.default_rng(3)
res = []
for m in (1, 3, 6):
    for rep in range(10):
        S = [(Fr(1), Fr(0)), (Fr(-1), Fr(0)), (Fr(0), Fr(1)), (Fr(0), Fr(-1))]
        n = 4 + 1 + m; c = 4; cl = list(range(5, n))
        adj = {v: [] for v in range(n)}
        def add(a, b): adj[a].append(b); adj[b].append(a)
        for s in range(4): add(s, c)
        for a in cl: add(a, c)
        for a in cl:
            for b in cl:
                if a < b: add(a, b)
        U = [c] + cl
        x = S + [(Fr(int(rng.integers(-30, 31)), 1000), Fr(int(rng.integers(-30, 31)), 1000))
                 for _ in range(1 + m)]
        Ep = [(a, b) for a in adj for b in adj[a] if a < b]
        pat = lambda z: frozenset(e for e in Ep if d2(z[e[0]], z[e[1]]) <= 1)
        p, sw = pat(x), []
        for t in range(1, 121):
            x = step(x, adj, U); q = pat(x)
            if q != p: sw.append(t)
            p = q
        res.append((m, len(sw), sw[-1] if sw else 0, tuple(float(v) for v in x[c])))
for r in res: print(r)
print("max latest switch:", max(r[2] for r in res))
