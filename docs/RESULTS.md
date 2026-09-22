# Measured results

Produced by `scripts/04_eval.sh` and `scripts/06_thinking_bench.sh` on this
machine:

* 2x Quadro GV100 (32 GB each), Xeon Gold 5222, 125 GB RAM
* `Qwen3.8-4B-Distill` Q8_0 (4.61 GB) + `Qwen3.5-4B` mmproj BF16 (0.68 GB)
* llama.cpp CUDA (sm_70), `-ngl 99`, ctx 16384, 4 parallel slots
* thinking strategy `off` (auto-selected: template `enable_thinking=false`)

## Decision quality

| suite | metric | result | notes |
|---|---|---|---|
| authored decisions (18 rows) | accuracy | **0.833** (15/18) | 194 ms p50 |
| animals (10 photos, 4-way) | accuracy | **1.000** (10/10) | 573 ms p50 |
| indexing, photos (6 x 3 shuffled orders) | item accuracy | **1.000** (18/18) | 3/3 perfect trials |
| indexing, mixed photos + texts | item accuracy | **0.667** (4/6) | cat/sheep swapped |
| indexing | assignment vs argmax | 0 changed | no collisions occurred |
| maze 5x5, legal moves, 3 seeds | success | **1.000** (3/3) | 0.854 step optimality |
| maze 5x5, all 4 directions | success | 0.000 (0/5) | cannot read local walls |

Missed decision rows: `incident-severity`, `code-review-risk`,
`irrelevance-check`.

## Mixed-input indexing

Inputs are an ordered list of images and/or texts; items are assigned to them
by one JEV decision per item (the caller supplies one `prompt` string with the
rules and the question, containing `{item}` - the engine adds no phrasing). The
per-item probabilities form an `items x inputs` matrix, and the final answer is
the **one-to-one assignment** that maximises `sum(log P)` (Hungarian algorithm,
`scipy.optimize.linear_sum_assignment`). Independent per-item argmax can map
two items onto the same input; the assignment cannot.

* photos only: **18/18** items correct across 3 different shuffled orders
  (and different question orders), 3/3 perfect trials
* mixed 3 photos + 3 text descriptions: **4/6**. Both misses are one swap
  (cat and sheep photos), i.e. a perception error, not an assignment error.
* assignment changed 0 answers here because the model was confident and
  collision-free; the unit tests cover the collision case explicitly
* image prefill is shared: after the first decision, each further decision
  reuses ~1.32k cached tokens (see `cached_tokens` per decision)

The same Hungarian solve is available for ordinary decisions:
`/v1/decide/batch` with `"assignment": "hungarian"` requires all items to share
one option set and returns a global matching (plus each item's independent
`argmax_choice`). Default `assignment: "none"` keeps decisions independent,
because arbitrary criteria may legitimately share an answer; a single
`/v1/decide` is the 1xN case where assignment is just argmax.

**Ordering is indexing.** There is no separate ordering API: make the inputs
rank-slot texts ("rank 1 = most agitated", ...), pass the reviews as items and
an explicit prompt saying rank 1 is the most agitated. The same
assignment then produces the permutation. `run_ordering_as_indexing` and the
`test_ordering_as_indexing` live test exercise exactly this path (bijection
plus most-agitated-first verified on the device).

## Prefill latency and memory

| prompt tokens | prefill ms | tok/s | peak server RSS | peak GPU |
|---|---|---|---|---|
| 127 | 102 | 1249 | 9924 MiB | 3773 + 3215 MiB |
| 231 | 156 | 1479 | 9880 MiB | 3773 + 3215 MiB |
| 644 | 242 | 2663 | 9944 MiB | 3773 + 3215 MiB |
| 1196 | 331 | 3617 | 9944 MiB | 3773 + 3215 MiB |

Memory is flat across prompt sizes: nothing is decoded, so no KV cache grows
across decisions. The limit is prefill time, not memory. A single-image
decision is ~200 ms; a 6-image indexing decision that reuses the image prefix
is similar.

## Reasoning-skipping benchmark (18 decisions + 10 images)

| strategy | boundary ok | think detected | letter top-1 | option mass | decisions | animals | p50 ms |
|---|---|---|---|---|---|---|---|
| `auto` (= `off`) | - | - | - | - | **0.833** | 1.000 | 202 |
| `off` | yes | no | yes | **0.998** | **0.833** | 1.000 | 202 |
| `answer_cue` | yes | no | yes | 0.652 | **0.833** | 1.000 | 268 |
| `system_no_reasoning` | yes | no | yes | 0.998 | 0.778 | 1.000 | 217 |
| `no_think` | yes | no | yes | 0.508 | 0.778 | 1.000 | 261 |
| `think_empty_cue` | yes | **yes** | **no** | 0.449 | 0.833 | 1.000 | 227 |
| `think_skip_cue` | yes | **yes** | **no** | 0.422 | 0.833 | 1.000 | 256 |

### Findings

1. **The template-level bypass is enough.** With `enable_thinking=false`, the
   chat template emits a closed `<think>\n\n</think>` block and the model's
   next token is an option letter (99.8% of the top-16 mass on the option
   slots, no thinking marker). The "reasoning distill always opens `<think>`"
   failure did not reproduce here.
2. **Injecting `<think>...</think>` text backfires.** `think_skip_cue` and
   `think_empty_cue` pass the boundary check, but the model continues with a
   fresh `<think>` opener anyway; the top-1 token stops being an option letter
   and the option mass drops from ~1.0 to ~0.44. Grammar still forces a letter,
   so accuracy survives, but the confidence signal is much weaker.
3. **Boundary validation earns its keep.** Cues ending in `:` merge with the
   following option letter in this tokenizer; the engine rejects them
   (`boundary_ok=false`) instead of reading a corrupted slot.

## Maze analysis

* The first maze runs failed because the renderer drew grid lines at cell
  centers instead of boundaries (the left column was half clipped). Fixed.
* Adding the blue visited-trail plus the explicit text state (borders,
  agent/goal description, ordered trajectory, visited cells) turned 5x5 mazes
  into 3/3 successes with 0.85 step optimality.
* In "all four directions" mode the model cannot read local walls: it chases a
  global bottom-right gradient and walks into walls (20-59 invalid moves).
  Direct wall probes ("is there a wall directly below the red circle?") score
  3/4, with the open-below case consistently misread. That is a perception
  limit of the 4B checkpoint at ~500 vision tokens, not a harness bug.

## Reproduce

```bash
./scripts/05_fetch_assets.sh
./scripts/04_eval.sh --suites all --thinking-bench --mazes 3
./scripts/06_thinking_bench.sh
```

Reports land in `results/<timestamp>/report.md`,
`results/<timestamp>/thinking-bench.md` and raw JSON next to them.

Per-task evidence is written under `results/artifacts/<task>/` and wiped at the
start of every run (in-place replacement, no stale accumulation):

* `decisions/` - `decisions.jsonl` with the exact rendered prompt, option
  probabilities and timings, plus `misses.md`;
* `animals/` - `contact-sheet.png` with expected/chosen burned in;
* `indexing/` - numbered contact sheets per trial (`trial-N-kind-order.png`),
  `trial-N-kind-assignments.txt`, `trials.json`;
* `maze/` - `seed-N/step-XX.png`, `final.png` and a per-seed contact sheet;
* `latency/` - `sweep.json`, `sweep.csv`;
* `thinking_bench/` - `thinking-bench.{json,md}` plus per-strategy reruns.
