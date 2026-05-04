//! `vpm-ledger`: append-only ledger Λ, trace DAG, and provenance graph.
//!
//! Realises:
//!
//! - The ledger schema `Λ = (id, type, mode, scope, ctx, sem, parents,
//!   entail, src, rebut, real, auth, prov, split, gen, dep, shift,
//!   verifiers, cert, cost, r, taint, frontier, time, hash)` (eq. 3 of
//!   `docs/architecture/01-contract-ledger-invariants.md`).
//! - The content-addressed trace DAG `τ = (N, E)` with node payload
//!   `B_n, c_n, M_n, …, Front_n, r_n` and edge payload
//!   `(a_e, o_e, cost_e, r_e, Cert_e, …, front_e)` (eq. 140 of
//!   `docs/architecture/08-training-system.md`).
//! - The provenance graph used by the cross-fitting split policy and the
//!   support guard (`docs/architecture/05-compiler-posterior.md` and
//!   `docs/architecture/08-training-system.md`).
//!
//! Invariant: every node and edge is keyed by its blake3 content hash, so
//! "every parse, source set, rebuttal set, entailment witness, verifier
//! result, realization check, dependence block, frontier measurement, or
//! risk estimate is computed once per hash and reused by every head whose
//! split policy permits it" (§8, structural-efficiency paragraph).

#![cfg_attr(not(test), forbid(unsafe_code))]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use vpm_core::{HashId, Mode, RiskVector, SemanticAtom};

/// Ledger row category for the executable MVP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryType {
    /// User/environment observation.
    Observation,
    /// Compiler or canonicalization event.
    Compile,
    /// DSL execution step.
    Execution,
    /// Verifier output.
    Verification,
    /// Gate decision.
    Gate,
    /// Memory admission/demotion event.
    Memory,
    /// Renderer output.
    Render,
}

/// Append-only ledger row Λ for the MVP fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// Monotonic row id.
    pub id: u64,
    /// Row category.
    pub entry_type: EntryType,
    /// Certified/soft/refusal mode.
    pub mode: Mode,
    /// Contract scope.
    pub scope: String,
    /// Context references.
    pub ctx: Vec<HashId>,
    /// Semantic atoms attached to this event.
    pub sem: Vec<SemanticAtom>,
    /// Parent ledger hashes.
    pub parents: Vec<HashId>,
    /// Authority labels.
    pub auth: Vec<String>,
    /// Certificate score contributed or consumed by this row.
    pub cert: f64,
    /// Execution/verification cost.
    pub cost: f64,
    /// Residual risk vector.
    pub risk: RiskVector,
    /// Taint labels inherited by downstream rows.
    pub taint: Vec<String>,
    /// Deterministic logical timestamp.
    pub time: u64,
    /// BLAKE3 content hash over all immutable row fields except id/time.
    pub hash: HashId,
}

/// Data required to append a row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerDraft {
    /// Row category.
    pub entry_type: EntryType,
    /// Certified/soft/refusal mode.
    pub mode: Mode,
    /// Contract scope.
    pub scope: String,
    /// Context references.
    pub ctx: Vec<HashId>,
    /// Semantic atoms.
    pub sem: Vec<SemanticAtom>,
    /// Parent ledger hashes.
    pub parents: Vec<HashId>,
    /// Authority labels.
    pub auth: Vec<String>,
    /// Certificate score.
    pub cert: f64,
    /// Cost.
    pub cost: f64,
    /// Risk.
    pub risk: RiskVector,
    /// Taint labels.
    pub taint: Vec<String>,
}

impl LedgerDraft {
    /// Create a draft with common default fields.
    pub fn new(entry_type: EntryType, scope: impl Into<String>) -> Self {
        Self {
            entry_type,
            mode: Mode::Soft,
            scope: scope.into(),
            ctx: Vec::new(),
            sem: Vec::new(),
            parents: Vec::new(),
            auth: vec!["data".to_owned()],
            cert: 0.0,
            cost: 0.0,
            risk: RiskVector::zero(),
            taint: Vec::new(),
        }
    }
}

/// Append-only, content-addressed ledger.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Ledger {
    entries: Vec<LedgerEntry>,
    #[serde(skip)]
    index: HashMap<HashId, usize>,
}

impl Ledger {
    /// Create an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a row, reusing an existing row when the content hash matches.
    pub fn append(&mut self, draft: LedgerDraft) -> LedgerEntry {
        let hash = HashId::of(&draft);
        if let Some(existing) = self.index.get(&hash) {
            return self.entries[*existing].clone();
        }
        let id = self.entries.len() as u64;
        let entry = LedgerEntry {
            id,
            entry_type: draft.entry_type,
            mode: draft.mode,
            scope: draft.scope,
            ctx: draft.ctx,
            sem: draft.sem,
            parents: draft.parents,
            auth: draft.auth,
            cert: draft.cert,
            cost: draft.cost,
            risk: draft.risk,
            taint: draft.taint,
            time: id,
            hash,
        };
        self.index.insert(entry.hash.clone(), self.entries.len());
        self.entries.push(entry.clone());
        entry
    }

    /// All ledger rows in append order.
    pub fn entries(&self) -> &[LedgerEntry] {
        &self.entries
    }

    /// Last row when present.
    pub fn last(&self) -> Option<&LedgerEntry> {
        self.entries.last()
    }

    /// Rebuild the hash index after deserialization.
    pub fn rebuild_index(&mut self) {
        self.index = self
            .entries
            .iter()
            .enumerate()
            .map(|(idx, row)| (row.hash.clone(), idx))
            .collect();
    }

    /// Compact report for CLI/API surfaces.
    pub fn summary(&self) -> LedgerSummary {
        LedgerSummary {
            entries: self.entries.len(),
            total_cost: self.entries.iter().map(|entry| entry.cost).sum(),
            total_cert: self.entries.iter().map(|entry| entry.cert).sum(),
            total_risk: self
                .entries
                .iter()
                .fold(RiskVector::zero(), |acc, row| acc.plus(row.risk)),
        }
    }
}

/// Small ledger summary used in reports.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LedgerSummary {
    /// Number of rows.
    pub entries: usize,
    /// Sum of costs.
    pub total_cost: f64,
    /// Sum of row certificate scores.
    pub total_cert: f64,
    /// Componentwise cumulative risk.
    pub total_risk: RiskVector,
}

/// Trace DAG node keyed by payload hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceNode {
    /// Node id/hash.
    pub id: HashId,
    /// Human-readable payload.
    pub payload: String,
    /// Ledger row hash that created this node.
    pub ledger_hash: HashId,
}

/// Trace DAG edge with action/observation labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceEdge {
    /// Parent node.
    pub from: HashId,
    /// Child node.
    pub to: HashId,
    /// Operator or action label.
    pub action: String,
    /// Cost charged by this transition.
    pub cost: f64,
}

/// Content-addressed trace DAG for one execution.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TraceDag {
    /// Nodes in insertion order.
    pub nodes: Vec<TraceNode>,
    /// Edges in insertion order.
    pub edges: Vec<TraceEdge>,
}

impl TraceDag {
    /// Add a node and return its hash.
    pub fn add_node(&mut self, payload: impl Into<String>, ledger_hash: HashId) -> HashId {
        let payload = payload.into();
        let id = HashId::of(&(payload.as_str(), &ledger_hash));
        if !self.nodes.iter().any(|node| node.id == id) {
            self.nodes.push(TraceNode {
                id: id.clone(),
                payload,
                ledger_hash,
            });
        }
        id
    }

    /// Add an edge.
    pub fn add_edge(&mut self, from: HashId, to: HashId, action: impl Into<String>, cost: f64) {
        self.edges.push(TraceEdge {
            from,
            to,
            action: action.into(),
            cost,
        });
    }
}

/// Parent links needed for split and support checks.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceGraph {
    links: HashMap<HashId, Vec<HashId>>,
}

impl ProvenanceGraph {
    /// Record parent hashes for a child hash.
    pub fn record(&mut self, child: HashId, parents: Vec<HashId>) {
        self.links.insert(child, parents);
    }

    /// Return true when `ancestor` is a direct parent of `child`.
    pub fn has_direct_parent(&self, child: &HashId, ancestor: &HashId) -> bool {
        self.links
            .get(child)
            .is_some_and(|parents| parents.iter().any(|parent| parent == ancestor))
    }
}

#[cfg(test)]
mod tests {
    use super::{EntryType, Ledger, LedgerDraft};

    #[test]
    fn ledger_reuses_duplicate_content_hash() {
        let mut ledger = Ledger::new();
        let row_a = ledger.append(LedgerDraft::new(EntryType::Observation, "test"));
        let row_b = ledger.append(LedgerDraft::new(EntryType::Observation, "test"));
        assert_eq!(row_a.hash, row_b.hash);
        assert_eq!(ledger.entries().len(), 1);
    }
}
