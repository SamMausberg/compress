#![no_main]
//! Fuzz target for the typed-DSL parser in `crates/vpm-dsl`.
//!
//! The DSL is the only path that turns external bytes into typed
//! mechanism state. A parser that crashes is a soundness bug because
//! the gate kernel cannot reject what it never sees. When the public
//! parser entry point lands (likely `vpm_dsl::parse_program`), wire it
//! up here.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Placeholder: drop the input and return. Replace with the real
    // parser as soon as one exists.
    let _ = data;
});
