# Evaluation modes

| Mode | Required tools | Chain stops when | Reported metrics |
| --- | --- | --- | --- |
| Default | Verus | Verus accepts | Verus Pass@k |
| With `--lynette` | Verus and the supplied Lynette checker | Verification and source-preservation checks pass | Verus and strict Pass@k |

The default mode records `safety_policy: verus_only`, safety as `disabled`, and
`strict_safe_pass` as null. Adding `--lynette` enables the safety checks; an
incompatible checker or failed tool invocation does not count as success.
RL training always requires safety checks.

## Metrics

An evaluation task has multiple independent repair chains. A chain succeeds if
an attempt passes the selected checks within the iteration budget. The initial
generation counts as the first attempt.

The evaluator reports the unbiased Pass@k estimator from the observed chains
and separately reports success among the first k chains. With three chains,
unbiased Pass@1 averages the three chain outcomes; Pass@3 records whether any
chain succeeds. Pass@1 here measures success after iterative repair.

Changing the stopping rule changes the evaluation protocol. The result
summarizer rejects comparisons with incompatible protocols. The reported
[paper results](../results/README.md) retain the paper's metric definitions.

## Optional checker

Build with `python scripts/prepare_safety_tool.py`. The checker compares
executable code, contracts, specification functions, original declarations,
and trust contexts. New trusted proof bypasses are rejected. Existing trusted
calls remain tied to their original proof and execution contexts. Some helper
renames, changed triggers, and proof changes involving existing trust are
conservatively rejected.

The source archive, patch, and licenses are in [third_party/](../third_party/README.md).
The build checks source hashes and reports `topoverus-trusted-ghost-context-v4`.
