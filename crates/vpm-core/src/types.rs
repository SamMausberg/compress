use serde::{Deserialize, Serialize};
use std::fmt;

/// Content-addressed identifier used by ledgers, traces, and witnesses.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HashId(pub String);

impl HashId {
    /// Hash any serializable value using canonical JSON bytes and BLAKE3.
    pub fn of<T: Serialize + ?Sized>(value: &T) -> Self {
        let bytes = serde_json::to_vec(value).expect("VPM values serialize to JSON");
        Self(blake3::hash(&bytes).to_hex().to_string())
    }
}

impl fmt::Display for HashId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Small typed value universe for the MVP executable kernel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Value {
    /// Signed integer value.
    Int(i64),
    /// UTF-8 text value.
    Text(String),
    /// Boolean value.
    Bool(bool),
}

impl Value {
    /// Return this value as an integer when its type matches.
    pub const fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(value) => Some(*value),
            Self::Text(_) | Self::Bool(_) => None,
        }
    }

    /// Return this value as text when its type matches.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value),
            Self::Int(_) | Self::Bool(_) => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(value) => write!(f, "{value}"),
            Self::Text(value) => f.write_str(value),
            Self::Bool(value) => write!(f, "{value}"),
        }
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

/// Ledgered semantic role for an atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AtomKind {
    /// User or environment observation.
    Observation,
    /// Claim that can be certified.
    Claim,
    /// Executable action request.
    Action,
    /// Clarifying question.
    Question,
    /// Program or verifier result.
    Result,
    /// Retrieved support source.
    Source,
    /// Retrieved contradiction or defeating source.
    Rebuttal,
    /// Active or archival memory write.
    Memory,
}

/// Rendering/execution mode partition used by the gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Certified atom in an executable or factual mode.
    Certified,
    /// Soft style/glue span without gated factual authority.
    Soft,
    /// Explicitly scoped assumption.
    Assumption,
    /// Refusal or abstention span.
    Refusal,
}

/// Minimal semantic atom shared by compiler, ledger, verifier, and renderer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticAtom {
    /// Stable content hash over the atom payload.
    pub id: HashId,
    /// Atom role.
    pub kind: AtomKind,
    /// Predicate or operator name.
    pub predicate: String,
    /// Typed arguments.
    pub args: Vec<Value>,
    /// Certified/soft/assumption/refusal mode.
    pub mode: Mode,
    /// Contract scope this atom belongs to.
    pub scope: String,
    /// Parent atom or ledger hashes.
    pub parents: Vec<HashId>,
}

impl SemanticAtom {
    /// Build a semantic atom and derive its stable id.
    pub fn new(
        kind: AtomKind,
        predicate: impl Into<String>,
        args: Vec<Value>,
        mode: Mode,
        scope: impl Into<String>,
        parents: Vec<HashId>,
    ) -> Self {
        #[derive(Serialize)]
        struct Payload<'a> {
            kind: AtomKind,
            predicate: &'a str,
            args: &'a [Value],
            mode: Mode,
            scope: &'a str,
            parents: &'a [HashId],
        }

        let predicate = predicate.into();
        let scope = scope.into();
        let payload = Payload {
            kind,
            predicate: &predicate,
            args: &args,
            mode,
            scope: &scope,
            parents: &parents,
        };
        Self {
            id: HashId::of(&payload),
            kind,
            predicate,
            args,
            mode,
            scope,
            parents,
        }
    }
}
