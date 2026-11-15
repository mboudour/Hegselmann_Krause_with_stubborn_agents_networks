"""Exact re-verification of two recent heterogeneous-HK counterexamples (complete graph, d = 1),
and of the fact that their endpoint agents are effectively stubborn:
(i)  Hegarty-Ognissanti-Wedin (arXiv:2610.03229, Sec. 2): convergent trajectory whose proximity
     digraph never freezes;
(ii) Etesami-Touri (arXiv:2610.20783, Thm 2): periodic orbit of period 10.
In both, replacing the endpoint radii by 0 (stubborn agents) leaves the trajectory unchanged,
while the remaining (regular) agents use at least two distinct bounds."""
from fractions import Fraction as Fr

def hk(x, r):
    n = len(x); y = []
    for i in range(n):
        nb = [j for j in range(n) if abs(x[j]-x[i]) <= r[i]]
        y.append(sum(x[j] for j in nb) / len(nb))
    return y
def pattern(x, r):
    n = len(x)
    return frozenset((i, j) for i in range(n) for j in range(n) if i != j and abs(x[j]-x[i]) <= r[i])

# (i) HOW, SBC model
x = [Fr(v) for v in (0, 38, 69, 84, 99, 130, 168)]
r = [Fr(v) for v in (18, 42, 48, 12, 48, 42, 18)]
r0 = [Fr(0)] + r[1:-1] + [Fr(0)]
xa, xb, ok, sw, prev = list(x), list(x), True, 0, None
for t in range(60):
    pred = [Fr(c) + Fr(-1, 6)**t * d for c, d in zip((0, 36, 72, 84, 96, 132, 168), (0, 2, -3, 0, 3, -2, 0))]
    ok &= (xa == pred) and (xa == xb)
    p = pattern(xa, r)
    if prev is not None and p != prev: sw += 1
    prev = p; xa, xb = hk(xa, r), hk(xb, r0)
print(f"HOW: closed form holds and stubborn endpoints give same trajectory for 60 steps: {ok}; "
      f"pattern changes: {sw} (one per step); regular bounds used: {sorted(set(map(str, r[1:-1])))}")

# (ii) Etesami-Touri
z = Fr(34576, 248589)
assert z == z/1024 + Fr(2161, 15552)
x0 = [Fr(0), z, Fr(2, 5), Fr(3, 5), 1 - z, Fr(1)]
r = [Fr(1, 1000), Fr(1, 4), Fr(59, 100), Fr(59, 100), Fr(1, 4), Fr(1, 1000)]
r0 = [Fr(0)] + r[1:-1] + [Fr(0)]
xa, xb, same = list(x0), list(x0), True
for t in range(1, 31):
    xa, xb = hk(xa, r), hk(xb, r0); same &= (xa == xb)
    if t in (10, 20, 30): print(f"ET: x({t}) == x(0): {xa == x0}")
print(f"ET: stubborn endpoints give same trajectory: {same}; period-10 orbit is non-constant: "
      f"{hk(x0, r) != x0}; regular bounds used: {sorted(set(map(str, r[1:-1])))}")
