#!/usr/bin/env python3
"""Direct SAT search for a <=k CCX round763 codec.

Free affine layers can be pushed to the final output side. Each Toffoli then
becomes an affine-conjugated shear:

    x -> x xor v * ((a.x xor a0) & (b.x xor b0))

with a,b independent and both annihilating v. A final free affine exists iff
the 27 output states lie in some affine hyperplane.
"""

from __future__ import annotations

import sys
from itertools import product

from z3 import And, Bool, BoolVal, Or, Solver, Xor, is_true, sat

N = 6


def xor_all(items):
    acc = BoolVal(False)
    for item in items:
        acc = Xor(acc, item)
    return acc


def dot(coeffs, bits):
    return xor_all([And(coeffs[i], bits[i]) for i in range(N)])


def reachable_states() -> list[list[bool]]:
    states: list[list[bool]] = []
    enc = [(0, 0), (1, 0), (1, 1)]
    for t0, t1, t2 in product(enc, repeat=3):
        states.append([bool(x) for x in [t0[0], t0[1], t1[0], t1[1], t2[0], t2[1]]])
    return states


def solve(k: int, timeout_ms: int) -> int:
    s = Solver()
    if timeout_ms:
        s.set("timeout", timeout_ms)

    gates = []
    for g in range(k):
        v = [Bool(f"g{g}_v_{i}") for i in range(N)]
        a = [Bool(f"g{g}_a_{i}") for i in range(N)]
        b = [Bool(f"g{g}_b_{i}") for i in range(N)]
        a0 = Bool(f"g{g}_a0")
        b0 = Bool(f"g{g}_b0")
        gates.append((v, a, b, a0, b0))

        s.add(Or(v))
        s.add(Or(a))
        s.add(Or(b))
        s.add(Or([a[i] != b[i] for i in range(N)]))
        s.add(dot(a, v) == BoolVal(False))
        s.add(dot(b, v) == BoolVal(False))

    outputs = []
    for state in reachable_states():
        bits = [BoolVal(bit) for bit in state]
        for v, a, b, a0, b0 in gates:
            qa = Xor(a0, dot(a, bits))
            qb = Xor(b0, dot(b, bits))
            q = And(qa, qb)
            bits = [Xor(bits[i], And(v[i], q)) for i in range(N)]
        outputs.append(bits)

    h = [Bool(f"h_{i}") for i in range(N)]
    hc = Bool("h_c")
    s.add(Or(h))
    for bits in outputs:
        s.add(Xor(hc, dot(h, bits)) == BoolVal(False))

    result = s.check()
    print(f"k={k} z3: {result}")
    if result != sat:
        return 1 if str(result) == "unknown" else 0

    model = s.model()

    def bv(x) -> int:
        return 1 if is_true(model.eval(x, model_completion=True)) else 0

    for gi, (v, a, b, a0, b0) in enumerate(gates):
        print(
            f"gate {gi}:",
            "v=", [bv(x) for x in v],
            "a=", [bv(x) for x in a],
            "a0=", bv(a0),
            "b=", [bv(x) for x in b],
            "b0=", bv(b0),
        )
    print("H =", [bv(x) for x in h], "hc =", bv(hc))
    return 0


def main() -> int:
    max_k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    timeout_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    code = 0
    for k in range(max_k + 1):
        rc = solve(k, timeout_ms)
        if rc:
            code = rc
        print()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
