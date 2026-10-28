import os
from pathlib import Path
import tempfile
import unittest

from topoverus.tools import safety, verify
from topoverus.trust import contract_projection, function_names, rust_tokens, trust_change


class TrustTests(unittest.TestCase):
    def test_helper_bypass(self):
        original = "fn f() ensures false, { }"
        candidate = "proof fn cheat() ensures false, { assume(false); } fn f() ensures false, { proof { cheat(); } }"
        self.assertIsNotNone(trust_change(original, candidate))

    def test_modified_assumption_and_verifier_annotations(self):
        self.assertIsNotNone(trust_change("assume(x > 0);", "assume(false);"))
        for marker in ("external_body", "external", "verify(false)"):
            self.assertIsNotNone(trust_change("", f"#[verifier::{marker}] proof fn cheat() {{}}"))
        self.assertIsNotNone(trust_change('', 'axiom fn cheat() ensures false;'))
        self.assertIsNotNone(trust_change('', 'proof(axiom) fn cheat() ensures false;'))

    def test_existing_benchmark_assumptions_remain_allowed(self):
        self.assertIsNone(trust_change("assume( x > 0 );", "assume(x>0);"))

    def test_comments_strings_and_lifetimes(self):
        original = '/* nested /* comment */ assume(false); */ fn f() { let s = r##"assume(false);"##; }'
        self.assertIsNone(trust_change("", original))
        self.assertIsNotNone(trust_change(original, original + " assume(false);"))
        tokens = rust_tokens("fn f<'a, 'b>(x: &'a str) { assume(false); }")
        self.assertIn("assume", tokens)

    def test_contract_projection_pins_original_headers_and_ignores_bodies(self):
        original = '#[verifier::external_body] pub proof fn h<T>(x: &\'a T) ensures true {}'
        names = function_names(original)
        self.assertEqual(names, {'h'})
        projected = contract_projection(original, names)
        self.assertIn('spec fn h<T>', projected)
        self.assertIn('external_body', projected)
        self.assertIn('topoverus_mode(proof)', projected)
        self.assertIn('{ unimplemented!() }', projected)
        self.assertEqual(projected, contract_projection(original.replace('{}', '{ assert(false); }'), names))
        self.assertNotEqual(projected, contract_projection(original.replace('ensures true', 'ensures false'), names))

    def test_projection_preserves_quantifiers_and_removes_new_untrusted_helpers(self):
        original = 'proof fn h() ensures forall |x: int| { x == x }, { }'
        names = function_names(original)
        projected = contract_projection(original, names)
        self.assertIn('forall |x: int| { x == x }', projected)
        candidate = original + ' #[inline] pub proof fn repair() {}'
        self.assertEqual(projected, contract_projection(candidate, names))
        self.assertEqual(projected, contract_projection(original + '\n/// New helper documentation\nproof fn repair() {}', names))
        self.assertNotEqual(projected, contract_projection(original + ' proof fn cheat() { assume(false); }', names))

    def test_projection_pins_assumptions_to_functions(self):
        original = 'proof fn h() { assume(false); } proof fn f() {}'
        names = function_names(original)
        candidate = 'proof fn h() {} proof fn f() { assume(false); }'
        self.assertNotEqual(contract_projection(original, names), contract_projection(candidate, names))

    def test_empty_main_before_verus_macro_keeps_macro_contents(self):
        original = 'fn main() {} verus! { proof fn h() ensures true {} }'
        projected = contract_projection(original, function_names(original))
        self.assertIn('spec fn main() { unimplemented!() } verus!', projected)
        self.assertIn('spec fn h() ensures true { unimplemented!() }', projected)
        self.assertIn('spec fn trusted() ensures false;', contract_projection(
            'pub axiom fn trusted() ensures false;', {'trusted'}))

    def test_proof_settings_do_not_change_contract_projection(self):
        original = 'pub fn f() ensures true {}'
        for attr in ('#[verifier::loop_isolation(false)]', '#[verifier::spinoff_prover]'):
            self.assertEqual(contract_projection(original, {'f'}), contract_projection(attr + original, {'f'}))


@unittest.skipUnless(os.environ.get("VERUS_BIN") and os.environ.get("LYNETTE_BIN"),
                     "Set VERUS_BIN and LYNETTE_BIN for real-tool trust regressions")
class TrustedContractIntegrationTests(unittest.TestCase):
    original = """use vstd::prelude::*;
verus! {
#[verifier::external_body]
proof fn helper() ensures true { }
fn target() ensures false { }
}
fn main() {}
"""

    def test_verifier_success_cannot_hide_changed_trusted_contract(self):
        candidate = self.original.replace("helper() ensures true", "helper() ensures false").replace(
            "fn target() ensures false { }", "fn target() ensures false { proof { helper(); } }")
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            self.assertEqual(verify(self.original, os.environ["VERUS_BIN"], work).status, "verification_failed")
            self.assertEqual(verify(candidate, os.environ["VERUS_BIN"], work).status, "verified")
            self.assertIsNone(trust_change(self.original, candidate))
            self.assertEqual(safety(self.original, candidate, os.environ["LYNETTE_BIN"], work).status, "unsafe")

    def test_existing_annotation_cannot_be_relocated_to_target(self):
        candidate = self.original.replace("#[verifier::external_body]\nproof fn helper", "proof fn helper").replace(
            "fn target()", "#[verifier::external_body]\nfn target()")
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(trust_change(self.original, candidate))
            self.assertEqual(safety(self.original, candidate, os.environ["LYNETTE_BIN"], Path(directory)).status, "unsafe")

    def test_ignored_body_change_and_normal_proof_repair_are_allowed(self):
        candidate = self.original.replace("helper() ensures true { }", "helper() ensures true { assert(false); }").replace(
            "fn target() ensures false { }", "fn target() ensures false { proof { helper(); } }")
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(safety(self.original, candidate, os.environ["LYNETTE_BIN"], Path(directory)).status, "safe")

    def test_existing_assumption_cannot_hide_changed_helper_contract(self):
        original = self.original.replace('#[verifier::external_body]\nproof fn helper() ensures true { }',
            'proof fn helper() requires false ensures false { assume(false); }')
        candidate = original.replace('requires false', 'requires true').replace(
            'fn target() ensures false { }', 'fn target() ensures false { proof { helper(); } }')
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            self.assertEqual(verify(candidate, os.environ["VERUS_BIN"], work).status, "verified")
            self.assertIsNone(trust_change(original, candidate))
            self.assertEqual(safety(original, candidate, os.environ["LYNETTE_BIN"], work).status, "unsafe")
            repair = original.replace('assume(false);', 'assume(false); assert(true);')
            self.assertEqual(safety(original, repair, os.environ["LYNETTE_BIN"], work).status, "safe")

    def test_macro_target_contract_and_executable_code_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            changed_contract = self.original.replace('target() ensures false', 'target() ensures true')
            self.assertEqual(safety(self.original, changed_contract, os.environ["LYNETTE_BIN"], work).status, 'unsafe')
            original = 'use vstd::prelude::*; verus! { fn f() -> u32 { 1 } } fn main() {}'
            self.assertEqual(safety(original, original.replace('{ 1 }', '{ 2 }'), os.environ["LYNETTE_BIN"], work).status, 'unsafe')

    def test_solver_settings_and_new_proof_helpers_remain_allowed(self):
        original = self.original.replace('fn target()', 'pub fn target()')
        candidate = original.replace('pub fn target()', '#[verifier::loop_isolation(false)] pub fn target()')
        candidate += '\n/// A new verified helper\nverus! { proof fn aux() ensures true {} }\n'
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(safety(original, candidate, os.environ["LYNETTE_BIN"], Path(directory)).status, 'safe')

    def test_existing_assumption_cannot_move_past_assignment(self):
        original = '''use vstd::prelude::*; verus! {
fn target(x: i32) -> (r: i32) requires x > 1, x < 10 ensures r > 1 {
    let mut y = x; assume(y > 1); y = y - 1; y
} } fn main() {}'''
        candidate = original.replace('assume(y > 1); y = y - 1;', 'y = y - 1; assume(y > 1);')
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            self.assertEqual(verify(original, os.environ['VERUS_BIN'], work).status, 'verification_failed')
            self.assertEqual(verify(candidate, os.environ['VERUS_BIN'], work).status, 'verified')
            self.assertIsNone(trust_change(original, candidate))
            self.assertEqual(safety(original, candidate, os.environ['LYNETTE_BIN'], work).status, 'unsafe')

    def test_existing_proof_assumption_cannot_become_unconditional(self):
        original = '''use vstd::prelude::*; verus! {
proof fn helper() ensures false { if false { assume(false); } }
fn target() ensures false { }
} fn main() {}'''
        candidate = original.replace('if false { assume(false); }', 'assume(false);').replace(
            'fn target() ensures false { }', 'fn target() ensures false { proof { helper(); } }')
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            self.assertEqual(verify(original, os.environ['VERUS_BIN'], work).status, 'verification_failed')
            self.assertEqual(verify(candidate, os.environ['VERUS_BIN'], work).status, 'verified')
            self.assertIsNone(trust_change(original, candidate))
            self.assertEqual(safety(original, candidate, os.environ['LYNETTE_BIN'], work).status, 'unsafe')

    def test_erased_branch_cannot_hide_executable_condition_effect(self):
        original = 'use vstd::prelude::*; verus! { fn target(v: &mut Vec<u32>) ensures true { } } fn main() {}'
        candidate = original.replace('ensures true { }', 'ensures true { if { v.push(0); true } { assert(true); } }')
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            self.assertEqual(verify(original, os.environ['VERUS_BIN'], work).status, 'verified')
            self.assertEqual(verify(candidate, os.environ['VERUS_BIN'], work).status, 'verified')
            self.assertEqual(safety(original, candidate, os.environ['LYNETTE_BIN'], work).status, 'unsafe')

    def test_conditional_proof_on_boolean_variable_remains_allowed(self):
        original = 'use vstd::prelude::*; verus! { fn target(flag: bool) ensures true { } } fn main() {}'
        candidate = original.replace('ensures true { }', 'ensures true { if flag { assert(true) by { } } }')
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            self.assertEqual(verify(candidate, os.environ['VERUS_BIN'], work).status, 'verified')
            self.assertEqual(safety(original, candidate, os.environ['LYNETTE_BIN'], work).status, 'safe')


if __name__ == "__main__":
    unittest.main()
