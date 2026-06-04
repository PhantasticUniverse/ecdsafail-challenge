#!/usr/bin/env python3
"""Search for a 3-CCX round763 codec via affine-layer SAT.

This models every circuit of the form

    A3 C A2 C A1 C A0

where C is a canonical Toffoli x2 ^= x0 & x1 and each Ai is an arbitrary
invertible affine transform over GF(2). Since arbitrary X/CX gates are free,
this is a complete normal form for three Toffolis separated by free affine
logic. A3 is represented by asking whether the post-third-CCX images of the
27 reachable trit states lie in any affine hyperplane.
"""

from __future__ import annotations

import sys
from itertools import product

from z3 import And, Bool, BoolVal, Or, Solver, Xor, is_true, sat

N = 6
K = 3


def xor_all(items):
    acc = BoolVal(False)
    for item in items:
        acc = Xor(acc, item)
    return acc


def reachable_states() -> list[list[bool]]:
    states: list[list[bool]] = []
    enc = [(0, 0), (1, 0), (1, 1)]
    for t0, t1, t2 in product(enc, repeat=3):
        bits = [t0[0], t0[1], t1[0], t1[1], t2[0], t2[1]]
        states.append([bool(b) for b in bits])
    return states


def affine_layer(prefix: str):
    m = [[Bool(f"{prefix}_m_{i}_{j}") for j in range(N)] for i in range(N)]
    c = [Bool(f"{prefix}_c_{i}") for i in range(N)]
    inv = [[Bool(f"{prefix}_inv_{i}_{j}") for j in range(N)] for i in range(N)]
    return m, c, inv


def constrain_invertible(s: Solver, m, inv) -> None:
    # m * inv == I over GF(2). For square matrices this is enough.
    for i in range(N):
        for k in range(N):
            terms = [And(m[i][j], inv[j][k]) for j in range(N)]
            s.add(xor_all(terms) == BoolVal(i == k))


def apply_affine(bits, m, c):
    out = []
    for i in range(N):
        terms = [And(m[i][j], bits[j]) for j in range(N)]
        out.append(Xor(c[i], xor_all(terms)))
    return out


def apply_ccx(bits):
    out = list(bits)
    out[2] = Xor(out[2], And(out[0], out[1]))
    return out


def rank(rows: list[list[int]]) -> int:
    rows = [row[:] for row in rows if any(row)]
    r = 0
    for col in range(N):
        pivot = next((i for i in range(r, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i][col]:
                rows[i] = [a ^ b for a, b in zip(rows[i], rows[r])]
        r += 1
    return r


def complete_final_affine(h: list[int], hc: int):
    rows: list[list[int]] = []
    for i in range(N):
        e = [0] * N
        e[i] = 1
        if len(rows) < N - 1 and rank(rows + [e, h]) == len(rows) + 2:
            rows.append(e)
    for i in range(N):
        if len(rows) == N - 1:
            break
        e = [0] * N
        e[i] = 1
        if rank(rows + [e]) == len(rows) + 1 and rank(rows + [e, h]) == len(rows) + 2:
            rows.append(e)
    assert len(rows) == N - 1, rows
    rows.append(h)
    offs = [0] * (N - 1) + [hc]
    assert rank(rows) == N
    return rows, offs


def reduce_to_identity_ops(m: list[list[int]]):
    a = [row[:] for row in m]
    ops: list[tuple[str, int, int]] = []
    for col in range(N):
        pivot = next((r for r in range(col, N) if a[r][col]), None)
        assert pivot is not None, (col, a)
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            ops.append(("SWAP", col, pivot))
        for r in range(N):
            if r != col and a[r][col]:
                a[r] = [x ^ y for x, y in zip(a[r], a[col])]
                ops.append(("CX", col, r))
    assert a == [[1 if i == j else 0 for j in range(N)] for i in range(N)]
    return ops


def synth_affine(m: list[list[int]], c: list[int]):
    # Row ops reduce M -> I. Reversing those self-inverse ops builds M from I.
    ops = list(reversed(reduce_to_identity_ops(m)))
    out: list[tuple] = []
    for op in ops:
        if op[0] == "SWAP":
            _, a, b = op
            out.extend([("CX", a, b), ("CX", b, a), ("CX", a, b)])
        else:
            out.append(op)
    for i, bit in enumerate(c):
        if bit:
            out.append(("X", i))
    return out


def apply_gate_int(v: int, gate: tuple) -> int:
    if gate[0] == "X":
        return v ^ (1 << gate[1])
    if gate[0] == "CX":
        _, c, t = gate
        return v ^ (((v >> c) & 1) << t)
    if gate[0] == "CCX":
        _, a, b, t = gate
        return v ^ ((((v >> a) & 1) & ((v >> b) & 1)) << t)
    raise ValueError(gate)


def verify_gate_path(gates: list[tuple]) -> None:
    ints = []
    for bits in reachable_states():
        v = sum((1 << i) for i, b in enumerate(bits) if b)
        for gate in gates:
            v = apply_gate_int(v, gate)
        ints.append(v)
    assert len(set(ints)) == len(ints)
    assert all(((v >> 5) & 1) == 0 for v in ints)


def main() -> int:
    s = Solver()
    timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if timeout:
        s.set("timeout", timeout)

    layers = [affine_layer(f"a{i}") for i in range(K)]
    for m, _c, inv in layers:
        constrain_invertible(s, m, inv)

    outputs = []
    for state in reachable_states():
        bits = [BoolVal(b) for b in state]
        for m, c, _inv in layers:
            bits = apply_affine(bits, m, c)
            bits = apply_ccx(bits)
        outputs.append(bits)

    h = [Bool(f"h_{i}") for i in range(N)]
    hc = Bool("h_c")
    s.add(Or(h))
    for bits in outputs:
        s.add(Xor(hc, xor_all([And(h[i], bits[i]) for i in range(N)])) == BoolVal(False))

    result = s.check()
    print("z3:", result)
    if result != sat:
        return 1 if str(result) == "unknown" else 0

    model = s.model()

    def bv(x) -> int:
        return 1 if is_true(model.eval(x, model_completion=True)) else 0

    gate_path: list[tuple] = []
    for idx, (m, c, _inv) in enumerate(layers):
        mi = [[bv(m[i][j]) for j in range(N)] for i in range(N)]
        ci = [bv(c[i]) for i in range(N)]
        print(f"A{idx}_M =", mi)
        print(f"A{idx}_c =", ci)
        gate_path.extend(synth_affine(mi, ci))
        gate_path.append(("CCX", 0, 1, 2))

    hv = [bv(x) for x in h]
    hcv = bv(hc)
    fm, fc = complete_final_affine(hv, hcv)
    print("H =", hv, "hc =", hcv)
    print("A3_M =", fm)
    print("A3_c =", fc)
    gate_path.extend(synth_affine(fm, fc))
    verify_gate_path(gate_path)

    ccx_count = sum(1 for g in gate_path if g[0] == "CCX")
    print("ccx_count =", ccx_count)
    print("gate_count =", len(gate_path))
    for gate in gate_path:
        print(gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
