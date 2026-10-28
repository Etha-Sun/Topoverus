import os
from pathlib import Path
import tempfile
import unittest

from topoverus.tools import safety, verify


@unittest.skipUnless(os.environ.get('VERUS_BIN') and os.environ.get('LYNETTE_BIN'),
                     'Real verifier and patched checker required')
class TrustedGhostContextTests(unittest.TestCase):
    def check_bypass(self, body, before, after):
        original = 'use vstd::prelude::*; verus! { ' + body + ' } fn main() {}'
        candidate = original.replace(before, after)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            self.assertEqual(verify(original, os.environ['VERUS_BIN'], work).status, 'verification_failed')
            self.assertEqual(verify(candidate, os.environ['VERUS_BIN'], work).status, 'verified')
            self.assertEqual(safety(original, candidate, os.environ['LYNETTE_BIN'], work).status, 'unsafe')

    def test_ghost_binding_cannot_make_existing_assumption_contradictory(self):
        self.check_bypass('fn target() ensures false { let ghost value: int = 0; assume(value == 0); }',
                          'value: int = 0', 'value: int = 1')

    def test_assertion_proof_cannot_remove_existing_assumption_guard(self):
        self.check_bypass('fn target() ensures false { assert(false) by { if false { assume(false); } } }',
                          'if false { assume(false); }', 'assume(false);')

    def test_ghost_assignment_cannot_change_later_assumption_value(self):
        self.check_bypass('fn target() ensures false { let ghost mut value: int = 0; proof { value = 0; } assume(value == 0); }',
                          'proof { value = 0; }', 'proof { value = 1; }')

    def test_ghost_guard_cannot_enable_existing_assumption(self):
        self.check_bypass('fn target() ensures false { let ghost flag: bool = false; proof { if flag { assume(false); } } }',
                          'flag: bool = false', 'flag: bool = true')

    def test_ghost_initializer_cannot_enable_existing_assumption(self):
        self.check_bypass('fn target() ensures false { let ghost value: int = { if false { assume(false); } 0 }; }',
                          'if false { assume(false); }', 'assume(false);')

    def test_quantified_assertion_cannot_remove_existing_assumption_guard(self):
        self.check_bypass('spec fn bad(x: int) -> bool { false } fn target() ensures true { assert forall |x: int| #[trigger] bad(x) by { if false { assume(false); } }; }',
                          'if false { assume(false); }', 'assume(false);')

    def test_plain_verified_assertion_with_existing_assumption_is_allowed(self):
        original = 'use vstd::prelude::*; verus! { fn target() ensures true { let ghost value: int = 0; assume(value == 0); } } fn main() {}'
        candidate = original.replace('assume(value == 0);', 'assume(value == 0); assert(value == 0);')
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            self.assertEqual(verify(candidate, os.environ['VERUS_BIN'], work).status, 'verified')
            self.assertEqual(safety(original, candidate, os.environ['LYNETTE_BIN'], work).status, 'safe')

    def test_ghost_proof_changes_without_existing_assumptions_are_allowed(self):
        original = 'use vstd::prelude::*; verus! { fn target() ensures true { let ghost value: int = 0; } } fn main() {}'
        candidate = original.replace('value: int = 0', 'value: int = 1')
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            self.assertEqual(verify(candidate, os.environ['VERUS_BIN'], work).status, 'verified')
            self.assertEqual(safety(original, candidate, os.environ['LYNETTE_BIN'], work).status, 'safe')
