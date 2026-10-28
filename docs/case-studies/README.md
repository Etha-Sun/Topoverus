# Paper case studies

These artifacts support Appendix B, **Qualitative Counterexample Case Studies**,
of *Curriculum Proof Repair with Learned Counterexamples*. The paper presents
shortened excerpts; the files here preserve the recorded code and counterexamples, including the
self-correction in Case A. Verifier source paths are shortened to filenames.

| Case | Benchmark task | Diagnosis | Historical generator |
| --- | --- | --- | --- |
| A | Dafnybench / `program_422_count_less_than` | Assertion occurs after `pop` and before `push` | `CTX-New-Dep` |
| B | Leetcode / `121-BestTimeSell` | Full-array claim should describe the processed prefix | `step_62_dep-reward-safe` |
| C | cleaned-verusbench / `task_id_447` | Loop context omits square and cube bounds | `CTX-New-Dep` |

## A: Assertion placement

Appendix B, page 15. Review record `V2-HCR-054`, sample `exp3v2-010`, chain 1,
iteration 1.

[Buggy code](a-assertion-placement/buggy.rs) ·
[Verus message](a-assertion-placement/verifier-error.txt) ·
[Recorded counterexample](a-assertion-placement/counterexample.txt)

For `numbers = [1,2,3]`, the intermediate state after the first `pop` has
`grow.len() + shrink.len() = 0 + 2`, while `numbers.len() = 3`.

## B: Prefix invariant

Appendix B, page 16. Constructed variant `best_time_sell_v2`, chain 1, iteration 1,
from the historical `best_time_sell_step62` run.

[Buggy code](b-prefix-invariant/buggy.rs) ·
[Verus message](b-prefix-invariant/verifier-error.txt) ·
[Recorded counterexample](b-prefix-invariant/counterexample.txt) ·
[Full model output](b-prefix-invariant/model-output.txt)

For `prices = [5,3,7]` at loop entry, the pair `j = 0, k = 2` requires
`7 - 5 <= max_pro = 0`. Restricting the invariant to `k < i` describes the
processed prefix.

## C: Arithmetic context

Appendix B, page 17. Review record `V2-HCR-051`, sample `exp3v2-053`, chain 2,
iteration 1.

[Buggy code](c-arithmetic-context/buggy.rs) ·
[Verus message](c-arithmetic-context/verifier-error.txt) ·
[Recorded counterexample](c-arithmetic-context/counterexample.txt)

The concrete input `nums = [10,20,30]` is within the arithmetic bounds. The
diagnosis concerns the facts available inside the loop: the invariant must carry
the square and cube bounds needed at the multiplication site.

## Sources

[sources.json](sources.json) records the original repository paths, sample and
review identifiers, historical generator labels, and SHA-256 hashes of the
included artifacts. Cases A and C come from the `exp3_v2` human-review records;
Case B comes from the saved motivated-example input and generation. These are
historical paper artifacts.

Benchmark code retains its source terms; see [licensing](../LICENSING.md).
