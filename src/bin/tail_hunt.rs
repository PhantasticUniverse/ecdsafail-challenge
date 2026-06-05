use alloy_primitives::U256;
use quantum_ecc::circuit::{analyze_ops, Op, OperationType, QubitOrBit};
use quantum_ecc::point_add::{self, SECP256K1_P};
use quantum_ecc::sim::Simulator;
use quantum_ecc::weierstrass_elliptic_curve::{sub_mod, WeierstrassEllipticCurve};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;

const NUM_TESTS: usize = 9024;
const NONCE_BITS: usize = 48;

fn secp256k1() -> WeierstrassEllipticCurve {
    WeierstrassEllipticCurve {
        modulus: SECP256K1_P,
        a: U256::from(0),
        b: U256::from(7),
        gx: U256::from_str_radix(
            "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
            16,
        )
        .unwrap(),
        gy: U256::from_str_radix(
            "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8",
            16,
        )
        .unwrap(),
        order: U256::from_str_radix(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
            16,
        )
        .unwrap(),
    }
}

#[derive(Clone, Copy, Debug)]
struct JacPoint {
    x: U256,
    y: U256,
    z: U256,
    infinity: bool,
}

impl JacPoint {
    fn infinity() -> Self {
        Self {
            x: U256::ZERO,
            y: U256::ZERO,
            z: U256::ZERO,
            infinity: true,
        }
    }

    fn affine(x: U256, y: U256) -> Self {
        Self {
            x,
            y,
            z: U256::from(1),
            infinity: false,
        }
    }
}

fn addm(a: U256, b: U256) -> U256 {
    a.add_mod(b, SECP256K1_P)
}

fn subm(a: U256, b: U256) -> U256 {
    sub_mod(a, b, SECP256K1_P)
}

fn mulm(a: U256, b: U256) -> U256 {
    a.mul_mod(b, SECP256K1_P)
}

fn sqrm(a: U256) -> U256 {
    mulm(a, a)
}

fn dblm(a: U256) -> U256 {
    addm(a, a)
}

fn jac_double(p: JacPoint) -> JacPoint {
    if p.infinity || p.y.is_zero() {
        return JacPoint::infinity();
    }
    let a = sqrm(p.x);
    let b = sqrm(p.y);
    let c = sqrm(b);
    let x_plus_b = addm(p.x, b);
    let d = dblm(subm(subm(sqrm(x_plus_b), a), c));
    let e = addm(addm(a, a), a);
    let f = sqrm(e);
    let x3 = subm(subm(f, d), d);
    let y3 = subm(mulm(e, subm(d, x3)), mulm(U256::from(8), c));
    let z3 = mulm(dblm(p.y), p.z);
    JacPoint {
        x: x3,
        y: y3,
        z: z3,
        infinity: false,
    }
}

fn jac_add_affine(p: JacPoint, q: (U256, U256)) -> JacPoint {
    if p.infinity {
        return JacPoint::affine(q.0, q.1);
    }
    if q.0.is_zero() && q.1.is_zero() {
        return p;
    }
    let z1z1 = sqrm(p.z);
    let u2 = mulm(q.0, z1z1);
    let s2 = mulm(q.1, mulm(p.z, z1z1));
    if p.x == u2 {
        if p.y == s2 {
            return jac_double(p);
        }
        return JacPoint::infinity();
    }
    let h = subm(u2, p.x);
    let hh = sqrm(h);
    let i = dblm(dblm(hh));
    let j = mulm(h, i);
    let r = dblm(subm(s2, p.y));
    let v = mulm(p.x, i);
    let x3 = subm(subm(sqrm(r), j), dblm(v));
    let y3 = subm(mulm(r, subm(v, x3)), mulm(dblm(p.y), j));
    let z3 = subm(subm(sqrm(addm(p.z, h)), z1z1), hh);
    JacPoint {
        x: x3,
        y: y3,
        z: z3,
        infinity: false,
    }
}

fn jac_add(p: JacPoint, q: JacPoint) -> JacPoint {
    if p.infinity {
        return q;
    }
    if q.infinity {
        return p;
    }
    let z1z1 = sqrm(p.z);
    let z2z2 = sqrm(q.z);
    let u1 = mulm(p.x, z2z2);
    let u2 = mulm(q.x, z1z1);
    let s1 = mulm(p.y, mulm(q.z, z2z2));
    let s2 = mulm(q.y, mulm(p.z, z1z1));
    if u1 == u2 {
        if s1 == s2 {
            return jac_double(p);
        }
        return JacPoint::infinity();
    }
    let h = subm(u2, u1);
    let i = sqrm(dblm(h));
    let j = mulm(h, i);
    let r = dblm(subm(s2, s1));
    let v = mulm(u1, i);
    let x3 = subm(subm(sqrm(r), j), dblm(v));
    let y3 = subm(mulm(r, subm(v, x3)), mulm(dblm(s1), j));
    let z3 = mulm(subm(subm(sqrm(addm(p.z, q.z)), z1z1), z2z2), h);
    JacPoint {
        x: x3,
        y: y3,
        z: z3,
        infinity: false,
    }
}

fn affine_batch3(
    a: JacPoint,
    b: JacPoint,
    c: JacPoint,
) -> Option<((U256, U256), (U256, U256), (U256, U256))> {
    if a.infinity || b.infinity || c.infinity {
        return None;
    }
    let ab = mulm(a.z, b.z);
    let abc = mulm(ab, c.z);
    let inv_abc = abc.inv_mod(SECP256K1_P)?;
    let inv_c = mulm(inv_abc, ab);
    let inv_ab = mulm(inv_abc, c.z);
    let inv_b = mulm(inv_ab, a.z);
    let inv_a = mulm(inv_ab, b.z);
    let ax = mulm(a.x, sqrm(inv_a));
    let bx = mulm(b.x, sqrm(inv_b));
    let cx = mulm(c.x, sqrm(inv_c));
    let ay = mulm(a.y, mulm(inv_a, sqrm(inv_a)));
    let by = mulm(b.y, mulm(inv_b, sqrm(inv_b)));
    let cy = mulm(c.y, mulm(inv_c, sqrm(inv_c)));
    Some(((ax, ay), (bx, by), (cx, cy)))
}

fn build_window_table(curve: &WeierstrassEllipticCurve) -> [[(U256, U256); 256]; 32] {
    let mut windows = [[(U256::ZERO, U256::ZERO); 256]; 32];
    let mut base = (curve.gx, curve.gy);
    for table in &mut windows {
        table[1] = base;
        for i in 2..256 {
            table[i] = curve.add(table[i - 1].0, table[i - 1].1, base.0, base.1);
        }
        for _ in 0..8 {
            base = curve.add(base.0, base.1, base.0, base.1);
        }
    }
    windows
}

fn scalar_mul_precomputed(windows: &[[(U256, U256); 256]; 32], k: U256) -> JacPoint {
    let mut acc = JacPoint::infinity();
    for (window, table) in windows.iter().enumerate() {
        let mut byte = 0usize;
        for bit_idx in 0..8 {
            if k.bit(8 * window + bit_idx) {
                byte |= 1usize << bit_idx;
            }
        }
        if byte != 0 {
            acc = jac_add_affine(acc, table[byte]);
        }
    }
    acc
}

fn absorb_op(hasher: &mut Shake256, op: &Op, q_target: Option<u64>) {
    hasher.update(&[op.kind as u8]);
    hasher.update(&op.q_control2.0.to_le_bytes());
    hasher.update(&op.q_control1.0.to_le_bytes());
    hasher.update(&q_target.unwrap_or(op.q_target.0).to_le_bytes());
    hasher.update(&op.c_target.0.to_le_bytes());
    hasher.update(&op.c_condition.0.to_le_bytes());
    hasher.update(&op.r_target.0.to_le_bytes());
}

fn prefix_hasher(ops: &[Op]) -> Shake256 {
    assert!(ops.len() >= 2 * NONCE_BITS);
    let tail_start = ops.len() - 2 * NONCE_BITS;
    for (idx, op) in ops[tail_start..].iter().enumerate() {
        assert_eq!(op.kind, OperationType::X, "tail must be fixed X;X pairs");
        assert_eq!(
            op.q_target.0, 0,
            "tail base op {idx} must target tx[0] before nonce override"
        );
    }
    let mut hasher = Shake256::default();
    hasher.update(b"quantum_ecc-fiat-shamir-v2");
    hasher.update(&(ops.len() as u64).to_le_bytes());
    for op in &ops[..tail_start] {
        absorb_op(&mut hasher, op, None);
    }
    hasher
}

fn xof_for_nonce(prefix: &Shake256, ops: &[Op], nonce: u64) -> sha3::Shake256Reader {
    let tail_start = ops.len() - 2 * NONCE_BITS;
    let mut hasher = prefix.clone();
    for i in 0..NONCE_BITS {
        let q = if (nonce >> i) & 1 == 1 { 1 } else { 0 };
        absorb_op(&mut hasher, &ops[tail_start + 2 * i], Some(q));
        absorb_op(&mut hasher, &ops[tail_start + 2 * i + 1], Some(q));
    }
    hasher.finalize_xof()
}

fn active_iterations() -> usize {
    std::env::var("DIALOG_GCD_ACTIVE_ITERATIONS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(259)
}

fn gcd_k2_enabled() -> bool {
    std::env::var("DIALOG_GCD_K2").ok().as_deref() == Some("1")
}

fn active_width(step: usize) -> usize {
    let slope = std::env::var("DIALOG_GCD_WIDTH_SLOPE_X1000")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s > 0.0)
        .map(|s| s / 1000.0)
        .unwrap_or(0.950);
    let margin = std::env::var("DIALOG_GCD_WIDTH_MARGIN")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|m| m.is_finite() && *m >= 0.0)
        .unwrap_or(25.0);
    let ideal = 256.0 - (step as f64) * slope + margin;
    let rounded = ((ideal.max(1.0) / 2.0).ceil() as usize) * 2;
    let lin = rounded.clamp(1, 256);
    let s = match std::env::var("DIALOG_GCD_WIDTH_BAND_TRIMS") {
        Ok(s) if !s.is_empty() => s,
        _ => return lin,
    };
    let trims: Vec<usize> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    if trims.is_empty() {
        return lin;
    }
    let iters = active_iterations();
    let band_size = ((iters + trims.len() - 1) / trims.len()).max(1);
    let band = (step / band_size).min(trims.len() - 1);
    let trim = trims[band];
    if trim == 0 {
        lin
    } else {
        ((lin.saturating_sub(trim) / 2) * 2).max(2)
    }
}

fn body_carry_band_trim(step: usize) -> Option<usize> {
    let s = std::env::var("DIALOG_GCD_BODY_CARRY_BAND_TRIMS").ok()?;
    if s.is_empty() {
        return None;
    }
    let trims: Vec<usize> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    if trims.is_empty() {
        return None;
    }
    let iters = active_iterations();
    let band_size = ((iters + trims.len() - 1) / trims.len()).max(1);
    let band = (step / band_size).min(trims.len() - 1);
    Some(trims[band])
}

fn body_carry_trunc_width(width: usize, step: usize) -> usize {
    let w = body_carry_band_trim(step).unwrap_or_else(|| {
        std::env::var("DIALOG_GCD_BODY_CARRY_TRUNC_W")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
    });
    width.saturating_sub(w).max(2)
}

fn compare_bits_for_step(step: usize, width: usize) -> usize {
    let global = std::env::var("DIALOG_GCD_COMPARE_BITS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(74)
        .min(width);
    if std::env::var("DIALOG_GCD_PA9024_COMPARE_SCHEDULE")
        .ok()
        .as_deref()
        != Some("1")
    {
        return global.max(1);
    }
    let schedule_margin = std::env::var("DIALOG_GCD_PA9024_COMPARE_SCHEDULE_MARGIN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(6);
    let scheduled = point_add::DIALOG_GCD_PA9024_COMPARE_SCHEDULE
        .get(step)
        .copied()
        .unwrap_or(global)
        .saturating_add(schedule_margin)
        .max(1)
        .min(width);
    scheduled.min(global).max(1)
}

fn high_nonzero(x: U256, width: usize) -> bool {
    width < 256 && !(x >> width).is_zero()
}

fn low_mask(width: usize) -> U256 {
    if width >= 256 {
        U256::MAX
    } else {
        (U256::from(1u64) << width) - U256::from(1u64)
    }
}

fn body_truncated_sub_ok(u: U256, v: U256, width: usize, step: usize) -> bool {
    let body_w = body_carry_trunc_width(width, step);
    if body_w >= width {
        return true;
    }
    if high_nonzero(u, body_w) {
        return false;
    }
    let mask = low_mask(body_w);
    (v & mask) >= (u & mask)
}

fn apply_clean_compare_bits() -> usize {
    std::env::var("DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(20)
        .clamp(1, 256)
}

fn high_slice(x: U256, bits: usize) -> U256 {
    x >> (256 - bits)
}

fn high_slice_not(x: U256, bits: usize) -> U256 {
    let mask = if bits == 256 {
        U256::MAX
    } else {
        (U256::from(1u64) << bits) - U256::from(1u64)
    };
    mask ^ high_slice(x, bits)
}

fn halvem(x: U256) -> U256 {
    if x.bit(0) {
        (x >> 1) + (SECP256K1_P >> 1) + U256::from(1u64)
    } else {
        x >> 1
    }
}

fn solinas_double_raw(x: U256) -> U256 {
    let c = U256::MAX
        .wrapping_sub(SECP256K1_P)
        .wrapping_add(U256::from(1u64));
    let y: U256 = x << 1;
    if x.bit(255) {
        y.wrapping_add(c)
    } else {
        y
    }
}

#[derive(Clone)]
struct GcdPrediction {
    log: Vec<(bool, bool, bool)>,
}

fn gcd_predict(mut u: U256, mut v: U256) -> Option<GcdPrediction> {
    let iters = active_iterations();
    let mut log = Vec::with_capacity(iters);
    for step in 0..iters {
        let width = active_width(step);
        if high_nonzero(u, width) || high_nonzero(v, width) {
            return None;
        }
        if !u.bit(0) {
            return None;
        }
        let compare_bits = compare_bits_for_step(step, width);
        let start = width - compare_bits;
        let trunc_gt = (u >> start) > (v >> start);
        let full_gt = u > v;
        let b0 = v.bit(0);
        if b0 && trunc_gt != full_gt {
            return None;
        }
        let b0_and_b1 = b0 && trunc_gt;
        if b0_and_b1 {
            std::mem::swap(&mut u, &mut v);
        }
        if b0 {
            if !body_truncated_sub_ok(u, v, width, step) {
                return None;
            }
            if v < u {
                return None;
            }
            v -= u;
        }
        v >>= 1;
        let shift2 = gcd_k2_enabled() && !v.bit(0);
        if shift2 {
            v >>= 1;
        }
        log.push((b0, b0_and_b1, shift2));
    }
    if u == U256::from(1) && v.is_zero() {
        Some(GcdPrediction { log })
    } else {
        None
    }
}

fn add_cleanup_ok(acc: U256, a: U256) -> Option<U256> {
    let c = U256::MAX
        .wrapping_sub(SECP256K1_P)
        .wrapping_add(U256::from(1u64));
    let sum = acc.wrapping_add(a);
    let overflow = sum < acc;
    let final_acc = if overflow { sum.wrapping_add(c) } else { sum };
    let bits = apply_clean_compare_bits();
    let trunc_overflow = high_slice(final_acc, bits) < high_slice(a, bits);
    if trunc_overflow == overflow {
        Some(final_acc)
    } else {
        None
    }
}

fn sub_cleanup_ok(acc: U256, a: U256) -> Option<U256> {
    let c = U256::MAX
        .wrapping_sub(SECP256K1_P)
        .wrapping_add(U256::from(1u64));
    let underflow = acc < a;
    let raw = acc.wrapping_sub(a);
    let final_acc = if underflow { raw.wrapping_sub(c) } else { raw };
    let bits = apply_clean_compare_bits();
    let trunc_not_underflow = high_slice(final_acc, bits) < high_slice_not(a, bits);
    if trunc_not_underflow == !underflow {
        Some(final_acc)
    } else {
        None
    }
}

fn predict_apply_product(log: &[(bool, bool, bool)], mut x: U256, mut y: U256) -> bool {
    for &(b0, b0_and_b1, shift2) in log.iter().rev() {
        y = solinas_double_raw(y);
        if shift2 {
            y = solinas_double_raw(y);
        }
        if b0 {
            let Some(next_y) = add_cleanup_ok(y, x) else {
                return false;
            };
            y = next_y;
        }
        if b0_and_b1 {
            std::mem::swap(&mut x, &mut y);
        }
    }
    true
}

fn predict_apply_quotient(log: &[(bool, bool, bool)], mut x: U256, mut y: U256) -> bool {
    for &(b0, b0_and_b1, shift2) in log {
        if b0_and_b1 {
            std::mem::swap(&mut x, &mut y);
        }
        if b0 {
            let Some(next_y) = sub_cleanup_ok(y, x) else {
                return false;
            };
            y = next_y;
        }
        y = halvem(y);
        if shift2 {
            y = halvem(y);
        }
    }
    true
}

fn nonce_fail_count(
    prefix: &Shake256,
    ops: &[Op],
    table: &[[(U256, U256); 256]; 32],
    nonce: u64,
    stop_after_first: bool,
) -> (usize, Option<usize>) {
    let mut xof = xof_for_nonce(prefix, ops, nonce);
    let mut failures = 0usize;
    let mut first = None;
    let mut shot = 0usize;
    for _ in 0..NUM_TESTS {
        let mut rb = [[0u8; 32]; 2];
        xof.read(&mut rb[0]);
        xof.read(&mut rb[1]);
        let k1 = U256::from_le_bytes(rb[0]);
        let k2 = U256::from_le_bytes(rb[1]);
        let t = scalar_mul_precomputed(table, k1);
        let o = scalar_mul_precomputed(table, k2);
        if t.infinity || o.infinity {
            continue;
        }
        let e = jac_add(t, o);
        let Some(((tx, ty), (ox, oy), (ex, _ey))) = affine_batch3(t, o, e) else {
            continue;
        };
        if tx == ox {
            continue;
        }
        let d1 = subm(tx, ox);
        let d2 = subm(ox, ex);
        let p1 = gcd_predict(SECP256K1_P, d1);
        let p2 = gcd_predict(SECP256K1_P, d2);
        let ok = if std::env::var("TAIL_HUNT_APPLY_CHECK").ok().as_deref() == Some("1") {
            let dy = subm(ty, oy);
            let lambda = match d1.inv_mod(SECP256K1_P) {
                Some(inv_d1) => mulm(dy, inv_d1),
                None => {
                    failures += 1;
                    first.get_or_insert(shot);
                    if stop_after_first {
                        return (failures, first);
                    }
                    shot += 1;
                    continue;
                }
            };
            p1.filter(|pred| predict_apply_quotient(&pred.log, U256::ZERO, dy))
                .is_some()
                && p2
                    .filter(|pred| predict_apply_product(&pred.log, lambda, U256::ZERO))
                    .is_some()
        } else {
            p1.is_some() && p2.is_some()
        };
        if !ok {
            failures += 1;
            first.get_or_insert(shot);
            if stop_after_first {
                return (failures, first);
            }
        }
        shot += 1;
    }
    (failures, first)
}

#[derive(Debug)]
struct ExactReport {
    ok: bool,
    classical_failures: usize,
    phase_garbage_batches: usize,
    ancilla_garbage_batches: usize,
}

fn nonce_exact_report(
    prefix: &Shake256,
    ops: &[Op],
    table: &[[(U256, U256); 256]; 32],
    nonce: u64,
) -> ExactReport {
    let mut xof = xof_for_nonce(prefix, ops, nonce);
    let mut targets = Vec::with_capacity(NUM_TESTS);
    let mut offsets = Vec::with_capacity(NUM_TESTS);
    let mut expected = Vec::with_capacity(NUM_TESTS);

    for _ in 0..NUM_TESTS {
        let mut rb = [[0u8; 32]; 2];
        xof.read(&mut rb[0]);
        xof.read(&mut rb[1]);
        let k1 = U256::from_le_bytes(rb[0]);
        let k2 = U256::from_le_bytes(rb[1]);
        let t = scalar_mul_precomputed(table, k1);
        let o = scalar_mul_precomputed(table, k2);
        if t.infinity || o.infinity {
            continue;
        }
        let e = jac_add(t, o);
        let Some(((tx, ty), (ox, oy), (ex, ey))) = affine_batch3(t, o, e) else {
            continue;
        };
        if tx == ox {
            continue;
        }
        targets.push((tx, ty));
        offsets.push((ox, oy));
        expected.push((ex, ey));
    }

    let (total_qubits, num_bits, _num_registers, layout_regs) = analyze_ops(ops.iter());
    assert!(layout_regs.len() >= 4);
    let mut sim = Simulator::new(total_qubits as usize, num_bits as usize, &mut xof);
    let mut report = ExactReport {
        ok: true,
        classical_failures: 0,
        phase_garbage_batches: 0,
        ancilla_garbage_batches: 0,
    };

    const BATCH: usize = 64;
    let num_batches = (targets.len() + BATCH - 1) / BATCH;
    for batch in 0..num_batches {
        let bs = BATCH.min(targets.len() - batch * BATCH);
        let cond_mask: u64 = if bs == 64 { u64::MAX } else { (1u64 << bs) - 1 };

        sim.clear_for_shot();
        for shot in 0..bs {
            let i = batch * BATCH + shot;
            sim.set_register(&layout_regs[0], targets[i].0, shot);
            sim.set_register(&layout_regs[1], targets[i].1, shot);
            sim.set_register(&layout_regs[2], offsets[i].0, shot);
            sim.set_register(&layout_regs[3], offsets[i].1, shot);
        }

        sim.apply_iter(ops.iter());

        for shot in 0..bs {
            let i = batch * BATCH + shot;
            let gx = sim.get_register(&layout_regs[0], shot);
            let gy = sim.get_register(&layout_regs[1], shot);
            if gx != expected[i].0 || gy != expected[i].1 {
                report.classical_failures += 1;
                report.ok = false;
            }
        }

        let phase = sim.phase & cond_mask;
        if phase != 0 {
            report.phase_garbage_batches += 1;
            report.ok = false;
        }

        for register in &layout_regs {
            for qb in register {
                if let QubitOrBit::Qubit(q) = *qb {
                    *sim.qubit_mut(q) = 0;
                }
            }
        }
        let mut garbage_q = false;
        for q in 0..total_qubits {
            if sim.qubit(quantum_ecc::circuit::QubitId(q)) & cond_mask != 0 {
                garbage_q = true;
                break;
            }
        }
        if garbage_q {
            report.ancilla_garbage_batches += 1;
            report.ok = false;
        }
    }

    report
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let start = args.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
    let count = args
        .get(2)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(1000);
    let step = args.get(3).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1);
    let threads = args
        .get(4)
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
    let full_count = std::env::var("TAIL_HUNT_FULL_COUNT").ok().as_deref() == Some("1");
    let skip_exact = std::env::var("TAIL_HUNT_SKIP_EXACT").ok().as_deref() == Some("1");

    std::env::set_var("DIALOG_TAIL_NONCE", "0");
    let ops = Arc::new(point_add::build());
    let prefix = Arc::new(prefix_hasher(&ops));
    let curve = secp256k1();
    let table = Arc::new(build_window_table(&curve));
    eprintln!(
        "tail_hunt ops={} start={} count={} step={} threads={}",
        ops.len(),
        start,
        count,
        step,
        threads
    );

    let found = Arc::new(AtomicBool::new(false));
    let global_best = Arc::new(Mutex::new((usize::MAX, 0usize, 0u64)));
    let mut handles = Vec::with_capacity(threads);

    for tid in 0..threads {
        let ops = Arc::clone(&ops);
        let prefix = Arc::clone(&prefix);
        let table = Arc::clone(&table);
        let found = Arc::clone(&found);
        let global_best = Arc::clone(&global_best);
        handles.push(thread::spawn(move || {
            let mut local_best = usize::MAX;
            let mut local_first = 0usize;
            let mut checked = 0u64;
            let mut i = tid as u64;
            while i < count && !found.load(Ordering::Relaxed) {
                let nonce = start + i * step;
                let (mut failures, mut first) =
                    nonce_fail_count(&prefix, ops.as_slice(), table.as_ref(), nonce, !full_count);
                if failures == 0 && !skip_exact {
                    let exact = nonce_exact_report(&prefix, ops.as_slice(), table.as_ref(), nonce);
                    if !exact.ok {
                        eprintln!("fast-clean nonce={nonce} rejected by exact check: {exact:?}");
                        failures = exact.classical_failures
                            + exact.phase_garbage_batches
                            + exact.ancilla_garbage_batches;
                        first = Some(0);
                    }
                }
                let first_score = first.unwrap_or(NUM_TESTS);
                if failures < local_best || (failures == local_best && first_score > local_first) {
                    local_best = failures;
                    local_first = first_score;
                    eprintln!(
                        "thread={tid} best nonce={nonce} failures={failures} first={first:?}"
                    );
                    let mut best = global_best.lock().unwrap();
                    if failures < best.0 || (failures == best.0 && first_score > best.1) {
                        *best = (failures, first_score, nonce);
                        eprintln!("global best nonce={nonce} failures={failures} first={first:?}");
                    }
                }
                if failures == 0 {
                    found.store(true, Ordering::SeqCst);
                    return Some(nonce);
                }
                checked += 1;
                if checked % 1000 == 0 {
                    eprintln!("thread={tid} progress checked={checked} local_best={local_best}");
                }
                i += threads as u64;
            }
            eprintln!("thread={tid} done checked={checked} local_best={local_best}");
            None
        }));
    }

    let mut found_nonce = None;
    for handle in handles {
        if let Some(nonce) = handle.join().unwrap() {
            found_nonce = Some(nonce);
        }
    }

    if let Some(nonce) = found_nonce {
        println!("FOUND {nonce}");
    } else {
        let best = global_best.lock().unwrap();
        println!(
            "NOT_FOUND best_nonce={} best_failures={} best_first={}",
            best.2, best.0, best.1
        );
    }
}
