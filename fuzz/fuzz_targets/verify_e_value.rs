#![no_main]
//! Fuzz target for the e-value runtime in `crates/vpm-verify`.
//!
//! The claim martingale (eq. 27) and the calibrated UCB (eq. 23) are
//! load-bearing for soundness. A crash inside the e-value combinator
//! is a soundness bug because raw pass counts cannot be promoted to
//! evidence by any other path. When `vpm_verify::EVal::combine_dep` (or
//! whatever the real entry point ends up being) lands, wire it up here.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Placeholder: drop the input and return.
    let _ = data;
});
