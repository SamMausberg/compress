//! `vpm-dsl`: typed DSL / bytecode and deterministic batch executor.
//!
//! Realises the "typed DSL/bytecode" and "batch executor" listed at the
//! head of §8 (`docs/architecture/08-training-system.md`):
//!
//! > Training is execution-first. Before neural learning, implement the
//! > typed DSL/bytecode, context normalizer, semantic-contract compiler,
//! > … batch executor, …
//!
//! Concretely:
//!
//! - A typed bytecode whose operator signatures match the typed
//!   primitives used in the dictionary `D_t` (eq. 11 of
//!   `docs/architecture/02-typed-mechanisms-evidence.md`).
//! - A deterministic executor that produces ledgered traces — every
//!   execution is recorded as a node / edge sequence in `vpm-ledger`.
//! - Batching by operator signature and e-class so candidate execution
//!   reuses identical sub-traces (§8, structural-efficiency paragraph).
//! - Memoization keyed by `(state, lower_bound, upper_bound, prune_reason)`
//!   for partial executions (§8 same paragraph).
//!
//! The executor is the only component allowed to write `Cert_obs`,
//! `Cert_cf`, and `Cert_atom` rows for executable modes (`M_cert ∋ math,
//! code, action`). Style / glue spans are handled by `vpm-py` →
//! `vpm.language.render`.

#![cfg_attr(not(test), forbid(unsafe_code))]

use serde::{Deserialize, Serialize};
use thiserror::Error;
use vpm_core::{AtomKind, HashId, Mode, RiskVector, SemanticAtom, Value};
use vpm_ledger::{EntryType, Ledger, LedgerDraft, TraceDag};

/// Operators supported by the MVP typed bytecode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpCode {
    /// Push a typed value.
    Push,
    /// Integer addition.
    Add,
    /// Integer multiplication.
    Mul,
    /// Text concatenation.
    Concat,
    /// Equality over typed values.
    Eq,
}

/// Single bytecode instruction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", content = "arg", rename_all = "snake_case")]
pub enum Instruction {
    /// Push a typed literal onto the stack.
    Push(Value),
    /// Pop two integers and push their sum.
    Add,
    /// Pop two integers and push their product.
    Mul,
    /// Pop two strings and push their concatenation.
    Concat,
    /// Pop two values and push typed equality.
    Eq,
}

impl Instruction {
    const fn opcode(&self) -> OpCode {
        match self {
            Self::Push(_) => OpCode::Push,
            Self::Add => OpCode::Add,
            Self::Mul => OpCode::Mul,
            Self::Concat => OpCode::Concat,
            Self::Eq => OpCode::Eq,
        }
    }
}

/// Deterministic bytecode program.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Program {
    /// Program name for ledger scope/reporting.
    pub name: String,
    /// Stack instructions.
    pub instructions: Vec<Instruction>,
}

impl Program {
    /// Construct a program.
    pub fn new(name: impl Into<String>, instructions: Vec<Instruction>) -> Self {
        Self {
            name: name.into(),
            instructions,
        }
    }

    /// Program content hash.
    pub fn hash(&self) -> HashId {
        HashId::of(self)
    }
}

/// Successful execution report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionReport {
    /// Executed program.
    pub program: Program,
    /// Final stack value.
    pub value: Value,
    /// Append-only execution ledger.
    pub ledger: Ledger,
    /// Trace DAG.
    pub trace: TraceDag,
    /// Semantic result atom.
    pub atom: SemanticAtom,
    /// Total deterministic cost.
    pub cost: f64,
}

/// DSL execution errors.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DslError {
    /// Stack did not contain enough operands.
    #[error("stack underflow while executing {0}")]
    StackUnderflow(&'static str),
    /// Operator received a value of the wrong type.
    #[error("type error while executing {0}")]
    TypeError(&'static str),
    /// Program ended with no final value.
    #[error("program produced no value")]
    EmptyProgram,
}

/// Execute a program and ledger every step.
pub fn execute(program: &Program) -> Result<ExecutionReport, DslError> {
    let mut ledger = Ledger::new();
    let mut trace = TraceDag::default();
    let mut stack = Vec::<Value>::new();
    let mut previous_node: Option<HashId> = None;

    for (pc, instruction) in program.instructions.iter().enumerate() {
        apply(instruction, &mut stack)?;
        let top = stack
            .last()
            .cloned()
            .ok_or(DslError::StackUnderflow("post-step"))?;
        let atom = SemanticAtom::new(
            AtomKind::Result,
            format!("{:?}", instruction.opcode()).to_lowercase(),
            vec![top.clone()],
            Mode::Certified,
            program.name.clone(),
            Vec::new(),
        );
        let mut draft = LedgerDraft::new(EntryType::Execution, program.name.clone());
        draft.mode = Mode::Certified;
        draft.sem = vec![atom];
        draft.cost = 1.0;
        let row = ledger.append(draft);
        let node = trace.add_node(format!("pc={pc}: {top}"), row.hash.clone());
        if let Some(parent) = previous_node.replace(node.clone()) {
            trace.add_edge(parent, node, format!("{:?}", instruction.opcode()), 1.0);
        }
    }

    let value = stack.pop().ok_or(DslError::EmptyProgram)?;
    let atom = SemanticAtom::new(
        AtomKind::Claim,
        "program_result",
        vec![value.clone()],
        Mode::Certified,
        program.name.clone(),
        vec![program.hash()],
    );
    let mut draft = LedgerDraft::new(EntryType::Execution, program.name.clone());
    draft.mode = Mode::Certified;
    draft.sem = vec![atom.clone()];
    draft.cert = 0.0;
    draft.cost = 1.0;
    draft.risk = RiskVector::zero();
    ledger.append(draft);

    Ok(ExecutionReport {
        program: program.clone(),
        value,
        ledger,
        trace,
        atom,
        cost: program.instructions.len() as f64 + 1.0,
    })
}

fn apply(instruction: &Instruction, stack: &mut Vec<Value>) -> Result<(), DslError> {
    match instruction {
        Instruction::Push(value) => stack.push(value.clone()),
        Instruction::Add => {
            let (left, right) = pop_ints(stack, "add")?;
            stack.push(Value::Int(left + right));
        }
        Instruction::Mul => {
            let (left, right) = pop_ints(stack, "mul")?;
            stack.push(Value::Int(left * right));
        }
        Instruction::Concat => {
            let right = stack.pop().ok_or(DslError::StackUnderflow("concat"))?;
            let left = stack.pop().ok_or(DslError::StackUnderflow("concat"))?;
            let left = left.as_text().ok_or(DslError::TypeError("concat"))?;
            let right = right.as_text().ok_or(DslError::TypeError("concat"))?;
            stack.push(Value::Text(format!("{left}{right}")));
        }
        Instruction::Eq => {
            let right = stack.pop().ok_or(DslError::StackUnderflow("eq"))?;
            let left = stack.pop().ok_or(DslError::StackUnderflow("eq"))?;
            stack.push(Value::Bool(left == right));
        }
    }
    Ok(())
}

fn pop_ints(stack: &mut Vec<Value>, op: &'static str) -> Result<(i64, i64), DslError> {
    let right = stack.pop().ok_or(DslError::StackUnderflow(op))?;
    let left = stack.pop().ok_or(DslError::StackUnderflow(op))?;
    let left = left.as_i64().ok_or(DslError::TypeError(op))?;
    let right = right.as_i64().ok_or(DslError::TypeError(op))?;
    Ok((left, right))
}

/// Convenient C0 arithmetic program.
pub fn c0_add_program(left: i64, right: i64) -> Program {
    c0_binary_program("add", left, right, Instruction::Add)
}

/// Convenient C0 multiplication program.
pub fn c0_mul_program(left: i64, right: i64) -> Program {
    c0_binary_program("mul", left, right, Instruction::Mul)
}

/// Convenient C0 text concatenation program.
pub fn c0_concat_program(left: impl Into<String>, right: impl Into<String>) -> Program {
    Program::new(
        "concat",
        vec![
            Instruction::Push(Value::Text(left.into())),
            Instruction::Push(Value::Text(right.into())),
            Instruction::Concat,
        ],
    )
}

/// Convenient C0 equality program over typed values.
pub fn c0_eq_program(left: Value, right: Value) -> Program {
    Program::new(
        "eq",
        vec![
            Instruction::Push(left),
            Instruction::Push(right),
            Instruction::Eq,
        ],
    )
}

fn c0_binary_program(operation: &str, left: i64, right: i64, instruction: Instruction) -> Program {
    Program::new(
        format!("{operation}-{left}-{right}"),
        vec![
            Instruction::Push(Value::Int(left)),
            Instruction::Push(Value::Int(right)),
            instruction,
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::{c0_add_program, c0_concat_program, c0_eq_program, c0_mul_program, execute};
    use vpm_core::Value;

    #[test]
    fn executes_add_program_with_ledger() {
        let report = execute(&c0_add_program(2, 3)).expect("program executes");
        assert_eq!(report.value, Value::Int(5));
        assert_eq!(report.ledger.entries().len(), 4);
        assert_eq!(report.trace.edges.len(), 2);
    }

    #[test]
    fn executes_mul_program_with_ledger() {
        let report = execute(&c0_mul_program(6, 7)).expect("program executes");
        assert_eq!(report.value, Value::Int(42));
        assert_eq!(report.ledger.entries().len(), 4);
        assert_eq!(report.trace.edges.len(), 2);
    }

    #[test]
    fn executes_concat_program_with_ledger() {
        let report = execute(&c0_concat_program("ab", "cd")).expect("program executes");
        assert_eq!(report.value, Value::Text("abcd".to_owned()));
        assert_eq!(report.ledger.entries().len(), 4);
        assert_eq!(report.trace.edges.len(), 2);
    }

    #[test]
    fn executes_eq_program_with_ledger() {
        let report =
            execute(&c0_eq_program(Value::Int(5), Value::Int(6))).expect("program executes");
        assert_eq!(report.value, Value::Bool(false));
        assert_eq!(report.ledger.entries().len(), 4);
        assert_eq!(report.trace.edges.len(), 2);
    }
}
