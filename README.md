# JEV Multimodal

Semantic-if decisions from **one forward pass** of a multimodal GGUF model.
No answer sentence, no decoding loop, no JSON to repair: you send one or more
images and/or a state plus typed options, and the engine reads the option
probabilities straight from the model's logits.

Decision types beyond a single multiple choice:
* **indexing** - assign items to an ordered list of mixed inputs (images and/or
  texts). You supply one `prompt` (rules + question, with `{item}`); the engine
  adds no task phrasing. The item<->input bijection is solved over the full
  probability matrix with SciPy's Hungarian algorithm (`item:index` output).
* **classification** - no wrapper: pass the labels as options to `decide` /
  `decide_many` (exactly what the animals suite does).

There is no separate ordering task by design: ordering is indexing where the
inputs are rank slots and the items are the texts (see
`indexing.run_ordering_as_indexing` and `tests/test_live.py`).

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (CPU **or** CUDA)
and `Qwen3.8-4B-Distill` (Q8_0) with the Qwen3.5 vision projector.

```
image + state + question + options
  -> chat template (reasoning-skipping strategy)
  -> token pinning + answer-boundary validation
  -> ONE forward pass, grammar over A/B/C/D, last-position probabilities
  => chosen option + per-option probabilities
```

## Measured results

On 2x Quadro GV100, Q8_0 + mmproj, auto strategy `off` (full tables in
[docs/RESULTS.md](docs/RESULTS.md)):

| suite | result | latency |
|---|---|---|
| authored decisions (18) | 0.833 accuracy | 194 ms p50 |
| animal photos (10) | 1.000 accuracy | 573 ms p50 |
| indexing, photos (6 x 3 shuffled orders) | 1.000 (18/18), 3/3 perfect | prefix-cached |
| indexing, mixed photos + texts | 0.667 (4/6) | prefix-cached |
| maze 5x5 (legal moves) | 3/3 success, 0.85 optimality | 312 ms p50 |
| maze 5x5 (all directions) | 0/5, cannot read local walls | |
| prefill 127 -> 1196 tokens | 102 -> 331 ms, flat 9.9 GB RSS | |

## Quick start (4 commands)

```bash
./scripts/00_install_deps.sh     # python deps (no torch needed)
./scripts/01_build_llama_cpp.sh  # llama.cpp with CUDA (or CPU)
cp .env.example .env             # add HF_TOKEN if you have one
./scripts/02_download_model.sh   # Qwen3.8-4B-Distill Q8_0 + vision mmproj
./scripts/03_run_api.sh          # API at http://127.0.0.1:8000/docs
```

Then:

```bash
./scripts/05_fetch_assets.sh     # animal photos for the vision eval
./scripts/04_eval.sh             # decisions + animals + maze + latency report
./scripts/06_thinking_bench.sh   # which reasoning-skipping strategy wins
```

## Why Q8_0 and not FP8?

There is no FP8 in GGUF. The format defines integer/block quants (`Q8_0`, ...),
`BF16`, `F16` and `MXFP4`/`NVFP4`, but no E4M3/E5M2 storage type, and the
upstream repo ships no FP8 file. `Q8_0` is the 8-bit option that exists: ~4.6 GB,
near-lossless for a 4B model, and it fits 6-8 GB cards. For bigger cards
`scripts/02b_convert_precision.sh` converts the full-precision safetensors to an
F16 GGUF (exact bf16->f16 round-trip; useful because Volta has FP16 tensor
cores but no BF16 path).

## Device selection

Everything is in `config.yaml`:

```yaml
device: auto        # auto | cpu | cuda | cuda:0 | cuda:0,1
gpu_layers: auto    # auto = all layers on GPU, none on CPU
tensor_split: null  # e.g. "0.5,0.5" for two GPUs
main_gpu: 0
threads: null       # null = physical cores
```

`device: auto` asks llama.cpp which devices exist and picks CUDA if present,
otherwise CPU. CPU mode passes `--device none -ngl 0`, so a CUDA build still
runs fully on the CPU. Every knob can also be set via `JEV_*` env vars
(`JEV_DEVICE=cpu`, `JEV_GPU_LAYERS=20`, ...).

## Reasoning models vs non-reasoning models

`Qwen3.8-4B-Distill` is distilled from chain-of-thought traces and will happily
open every answer with `<think>`, which breaks a logit readout. The engine
handles both families with named **reasoning-skipping strategies**:

| strategy | what it does |
|---|---|
| `off` | template-level `enable_thinking=false` only |
| `system_no_reasoning` | system prompt forbids reasoning |
| `think_skip_cue` | closed `<think>Skip internal reasoning.</think>` + answer cue |
| `think_empty_cue` | empty closed `<think>` block + answer cue |
| `answer_cue` | bare `Final answer:` cue |
| `no_think` | Qwen-style `/no_think` soft switch |

`thinking.mode: auto` probes them at startup and picks the first strategy that
both preserves the answer boundary and avoids a thinking opener. The choice is
empirical, so it is also benchmarked end-to-end:

```bash
./scripts/06_thinking_bench.sh
# -> results/<timestamp>/thinking-bench.md : accuracy + probe metrics
```

Measured on `Qwen3.8-4B-Distill` (see [docs/RESULTS.md](docs/RESULTS.md)):
template `enable_thinking=false` (`off`) already suppresses thinking
(option mass 0.998, no `<think>` opener). Injecting a closed
`<think>...</think>` block *induced* another `<think>` and collapsed the
option mass to ~0.44, so `off`/`answer_cue` are the default `auto_order`.
Cues ending in `:` were rejected outright by the boundary validator because
they merge with the following letter in the tokenizer. Pin any strategy in
`config.yaml` with `thinking.mode: off` (or another name) to skip the probe.

## HTTP API

Interactive docs: `http://127.0.0.1:8000/docs`. One example per endpoint.

**`GET /health`**

```bash
curl -s localhost:8000/health
# {"status":"ok"}
```

**`GET /v1/info`** - model, device, resolved thinking strategy, startup probes.

```bash
curl -s localhost:8000/v1/info
# {"model":".../Qwen3.8-4B-Q8_0.gguf","device":"cuda:all","thinking_strategy":"off",...}
```

**`POST /v1/decide`** - one decision over text and/or images. Images can be
added with `image_path`, `image_base64`, `images_paths`, `images_base64`.
Classification over labels is the same endpoint with the labels as options.

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "state": "Customer was charged twice for the same subscription.",
  "question": "Which queue should handle this request?",
  "options": [
    {"id": "billing", "description": "billing support"},
    {"id": "access",  "description": "account access support"},
    {"id": "shipping","description": "shipping support"}
  ]
}'
# {"chosen":"billing","confidence":0.993,"probabilities":{"billing":0.993,...},...}
```

**`POST /v1/decide/batch`** - many criteria over the same evidence; the shared
prefix is KV-cached, so later decisions only evaluate their tail. Add
`"assignment": "hungarian"` when the answers are a declared one-to-one
matching (all items must share the same option ids): choices are then solved
globally over the item x option probability matrix - the same solve as
`/v1/index` - and the response gains an `assignment` block. With the default
`"assignment": "none"` items are independent and may share an answer. A single
`/v1/decide` is the 1xN degenerate case, where assignment reduces to argmax.

```bash
curl -s localhost:8000/v1/decide/batch -H 'content-type: application/json' -d '{
  "state": "Checkout API returning 500 for 100% of requests in production for 12 minutes.",
  "items": [
    {"id": "severity", "question": "What is the severity?",
     "options": [{"id": "sev1", "description": "total outage"},
                 {"id": "sev2", "description": "degradation"},
                 {"id": "sev3", "description": "minor issue"}]},
    {"id": "page", "question": "Should on-call be paged now?",
     "options": [{"id": "yes", "description": "page now"},
                 {"id": "no", "description": "do not page"}]}
  ]
}'
# {"count":2,"results":[...],"cached_tokens_per_decision":[126,117]}
```

**`POST /v1/index`** - mixed image/text indexing. Inputs are ordered and
numbered 1..N; you write one `prompt` (rules + question with `{item}`). The
answer is a one-to-one assignment (`scipy.optimize.linear_sum_assignment`), so
two items can never land on the same input. Ordering uses the same call with
rank-slot inputs (CLI: `jev index --help`).

```bash
curl -s localhost:8000/v1/index -H 'content-type: application/json' -d '{
  "inputs": [
    {"type": "image", "image_path": "data/images/animals/horse.jpg", "label": "photo 1"},
    {"type": "text",  "text": "a photo of a green frog",               "label": "note 2"},
    {"type": "image", "image_path": "data/images/animals/cat.jpg",     "label": "photo 3"}
  ],
  "items": ["frog", "horse", "cat"],
  "prompt": "Assign every requested item to exactly one input that depicts or describes it; each input is used at most once. Which input matches the {item}?"
}'
# {"assignments":{"frog":2,"horse":1,"cat":3},"text":"frog:2\nhorse:1\ncat:3",...}
```

**`POST /v1/diagnose`** - raw next-token probes without grammar, per
reasoning-skipping strategy; shows whether the model opens with `<think>`.

```bash
curl -s localhost:8000/v1/diagnose -H 'content-type: application/json' \
  -d '{"strategies": ["off", "think_skip_cue", "on"]}'
# {"resolved_strategy":"off","probes":[{"strategy":"off","thinking_detected":false,...}],...}
```

**`POST /v1/maze/solve`** - runs the maze task end to end; step images and
contact sheets land in `results/artifacts/maze/`.

```bash
curl -s localhost:8000/v1/maze/solve -H 'content-type: application/json' \
  -d '{"width": 5, "height": 5, "seed": 1, "max_steps": 40}'
# {"success":true,"steps":20,"optimal_steps":10,"step_optimality":0.75,...}
```

**`POST /v1/bench/latency`** - prefill latency and peak memory vs prompt length.

```bash
curl -s localhost:8000/v1/bench/latency -H 'content-type: application/json' \
  -d '{"token_targets": [32, 128, 512], "repeats": 1}'
# {"rows":[{"target_tokens":32,"prompt_tokens":127,"prompt_ms":100.2,...},
#          {"target_tokens":128,"prompt_tokens":231,"prompt_ms":113.3,...},...]}
```

| endpoint | purpose |
|---|---|
| `GET /health` | liveness |
| `GET /v1/info` | model, device, resolved strategy, probes |
| `POST /v1/decide` | one decision (text and/or image) |
| `POST /v1/decide/batch` | many criteria over the same evidence (prefix cache) |
| `POST /v1/index` | mixed image/text indexing -> `item:index`, Hungarian assignment |
| `POST /v1/diagnose` | raw next-token probes for every strategy |
| `POST /v1/maze/solve` | run the maze task end to end |
| `POST /v1/bench/latency` | prefill latency vs prompt length |

## Tasks

* `decisions` - 18 authored labeled records across routing, moderation,
  incidents, code review, credit risk, logic, ...
* `animals` - Wikimedia Commons photos, 4-way (configurable) classification.
* `indexing` - photos-only trials (6 animal photos in shuffled orders) and a
  mixed trial (3 photos + 3 text descriptions in one input list). Every
  item->input decision feeds one probability matrix, and the final answer is
  the global assignment (no collisions possible). Scored against the shuffle.
  Ordering is demonstrated through this same call with rank-slot inputs.
* `maze` - perfect maze rendered as an image; black walls, red agent, green
  exit, and a blue trail of already visited cells. Every step is a JEV
  decision scored against the BFS-optimal direction. The prompt state spells
  out the borders, how the agent/goal/trail look, the visited cells and the
  ordered trajectory (`positions_in_order`) so the text carries the same
  information as the picture. Two action spaces:
  `legal_moves_only=true` (harness offers only open moves; tests planning) and
  `false` (all four directions; also tests wall perception).
  Images land in `data/images/generated/maze-*/`.
* `latency` - prefill time and peak RSS/GPU memory versus prompt length.

## Artifacts

Every task owns a directory under `results/artifacts/<task>/`. The directory is
wiped when the task starts, so re-running a task replaces stale outputs instead
of accumulating them:

```
results/artifacts/
  decisions/          decisions.jsonl (prompt + probabilities + timings), misses.md
  animals/            contact-sheet.png, decisions.jsonl
  indexing/           trial-N-kind-order.png + trial-N-kind-assignments.txt, trials.json
  maze/               seed-N/step-XX.png + final.png, seed-N-contact-sheet.png
  latency/            sweep.json, sweep.csv
  thinking_bench/     thinking-bench.{json,md} + per-strategy decisions/animals/maze
```

`summary.json` in each directory lists the files that run produced.

## Invariants enforced in code

* **Token pinning** - every option letter must encode to exactly one
  round-tripping token, and appending it to the prompt must extend the
  tokenization by exactly that token (`jev/engine.py:_pin`).
* **Single forward pass** - `n_predict=1`, grammar over the option letters,
  `post_sampling_probs=true`; no sampling loop ever runs.
* **Flat memory** - no KV cache growth across decisions; the latency suite
  records peak server RSS and GPU memory to prove it.
* **Auditability** - every decision returns `prompt_sha256`, token counts,
  timings, the grammar and the strategy used.

## Layout

```
config.yaml            runtime config (device, model, strategies, limits)
jev/
  engine.py            JEV readout + token pinning
  prompting.py         templates, messages, strategies
  client.py            llama-server HTTP client
  llama_server.py      process + device placement
  api.py / cli.py      FastAPI + CLI
  tasks/               decisions, animals, maze, evaluation
scripts/               setup / build / download / run / eval
third_party/llama.cpp  built from source by scripts/01
```

## Troubleshooting

* **`llama-server not found`** - run `scripts/01_build_llama_cpp.sh`.
* **`Unable to load model` / unknown architecture** - use the bundled
  llama.cpp build (Qwen3.5 hybrid attention needs a recent version).
* **`readout failed: no option-slot probability mass`** - check
  `POST /v1/diagnose`; usually the model is emitting a thinking token and the
  strategy needs to be `think_skip_cue` (or run the benchmark).
* **Image tasks error with "image input is not supported"** - the mmproj is
  missing; re-run `scripts/02_download_model.sh`.
