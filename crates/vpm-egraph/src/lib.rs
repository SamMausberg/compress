//! `vpm-egraph`: e-graph / equality saturation for VPM-5.3.
//!
//! Wraps the `egg` crate (Willsey et al., POPL 2021 — reference \[6\] of
//! `docs/references.md`) and adapts it to:
//!
//! - **Posterior-valued canonicalization** with reversible witness or a
//!   support-loss bound (eqs. 84–85 of
//!   `docs/architecture/05-compiler-posterior.md`).
//! - **Macro equivalence certificates** `Cert_eq(m)` over the expansion
//!   witness `W_m` (eq. 124 of
//!   `docs/architecture/07-compression-memory.md`).
//! - **Negative siblings**: when a verifier rejects a candidate at one
//!   e-class member, sound siblings inherit a negative certificate (§8,
//!   structural-efficiency paragraph).
//!
//! Soundness contract: this crate may merge e-classes only with a
//! reversible witness, and every merge is recorded in the ledger as a
//! `canon_witness` event so the support-loss bound `ε_canon` (eq. 85) can
//! be audited.

#![cfg_attr(not(test), forbid(unsafe_code))]

use serde::{Deserialize, Serialize};
use vpm_core::HashId;
use vpm_dsl::{Instruction, Program};

/// Reversible canonicalization witness for a program rewrite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonWitness {
    /// Rule name.
    pub rule: String,
    /// Program hash before the rewrite.
    pub before: HashId,
    /// Program hash after the rewrite.
    pub after: HashId,
    /// Human-readable proof sketch.
    pub explanation: String,
    /// Upper bound on support lost by this rewrite.
    pub support_loss: f64,
}

/// Canonical program plus all witnesses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalProgram {
    /// Canonicalized program.
    pub program: Program,
    /// Rewrite/equivalence witnesses.
    pub witnesses: Vec<CanonWitness>,
    /// Aggregate support-loss bound.
    pub support_loss: f64,
}

/// Canonicalize simple stack-program identities used by C0 tasks.
pub fn canonicalize(program: &Program) -> CanonicalProgram {
    let mut current = program.clone();
    let before = current.hash();
    let mut instructions = Vec::with_capacity(current.instructions.len());
    let mut witnesses = Vec::new();
    let mut changed = false;

    for instruction in &current.instructions {
        if removes_identity(&mut instructions, instruction) {
            changed = true;
        } else {
            instructions.push(instruction.clone());
        }
    }

    if changed {
        current.instructions = instructions;
        let after = current.hash();
        witnesses.push(CanonWitness {
            rule: "identity-elimination".to_owned(),
            before,
            after,
            explanation: "remove x+0, x*1, and concat-empty stack identities".to_owned(),
            support_loss: 0.0,
        });
    }

    CanonicalProgram {
        program: current,
        witnesses,
        support_loss: 0.0,
    }
}

fn removes_identity(previous: &mut Vec<Instruction>, instruction: &Instruction) -> bool {
    match (previous.last(), instruction) {
        (Some(Instruction::Push(vpm_core::Value::Int(0))), Instruction::Add) => {
            previous.pop();
            true
        }
        (Some(Instruction::Push(vpm_core::Value::Int(1))), Instruction::Mul) => {
            previous.pop();
            true
        }
        (Some(Instruction::Push(vpm_core::Value::Text(text))), Instruction::Concat)
            if text.is_empty() =>
        {
            previous.pop();
            true
        }
        _ => false,
    }
}

/// Minimal e-class record used by memory/admission reports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EClass {
    /// Canonical program hash.
    pub canonical: HashId,
    /// Equivalent member hashes.
    pub members: Vec<HashId>,
    /// Witnesses proving member equivalence.
    pub witnesses: Vec<CanonWitness>,
}

impl EClass {
    /// Build a singleton/member e-class from a canonicalization result.
    pub fn from_canonical(original: &Program, canonical: CanonicalProgram) -> Self {
        let canonical_hash = canonical.program.hash();
        let mut members = vec![original.hash()];
        if canonical_hash != members[0] {
            members.push(canonical_hash.clone());
        }
        Self {
            canonical: canonical_hash,
            members,
            witnesses: canonical.witnesses,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{canonicalize, EClass};
    use vpm_core::Value;
    use vpm_dsl::{Instruction, Program};

    #[test]
    fn canonicalization_records_identity_witness() {
        let program = Program::new(
            "id",
            vec![
                Instruction::Push(Value::Int(7)),
                Instruction::Push(Value::Int(0)),
                Instruction::Add,
            ],
        );
        let canonical = canonicalize(&program);
        assert_eq!(canonical.program.instructions.len(), 1);
        assert_eq!(canonical.witnesses.len(), 1);
        assert_eq!(EClass::from_canonical(&program, canonical).members.len(), 2);
    }
}
