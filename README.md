# LoRA fine-tuning for intent classification

**Fine-tuning a 0.5B-parameter open LLM with LoRA on a laptop, measured against zero-shot and few-shot prompting
of the same model.**

The task is [Banking77](https://huggingface.co/datasets/PolyAI/banking77): map a customer-support message from a
digital bank ("I am still waiting on my card?") to one of 77 fine-grained intents (`card_arrival`). The model
generates the intent label as text, so the same pipeline would work for any task with a closed answer set.

The question this project answers: for a small open model, how far do prompting tricks get you, and what does a
LoRA adapter (1.7% of the parameters, trained in one hour on an Apple M4 laptop) add on top?

`Python 3.11+` · `PyTorch` (MPS / CUDA / CPU) · `transformers` · `peft` (LoRA, QLoRA) · `Qwen2.5-0.5B-Instruct` ·
`typer` CLI · `pytest` · `ruff` · `Docker` · `GitHub Actions`

---

## Headline results

Banking77 test set, all 3,080 examples, greedy decoding. 95% confidence intervals from 1,000 bootstrap
resamples.

| System | Accuracy % (95% CI) | Macro-F1 % (95% CI) | Invalid outputs | Prompt tokens | Throughput (ex/s) |
|---|---|---|---|---|---|
| Zero-shot (label list in prompt) | 19.1 [17.7, 20.5] | 19.4 [17.9, 20.5] | 39.5% | 448 | 3.6 |
| Few-shot, 8 retrieved examples | 76.2 [74.8, 77.7] | 78.8 [77.2, 79.9] | 6.7% | 651 | 2.4 |
| **LoRA (r = 16), short prompt** | **92.9 [92.0, 93.8]** | **92.9 [92.0, 93.8]** | **0.2%** | **62** | **13.2** |

Paired bootstrap on the same 3,080 examples: LoRA beats few-shot by **+16.6 accuracy points (95% CI +15.2 to
+18.1)**; few-shot beats zero-shot by +57.1 (+55.3 to +59.0).

Throughput is batched greedy generation (batch 16) on the M4 GPU. The full table, including peak memory, is in
[`results/RESULTS.md`](results/RESULTS.md); every number comes from the JSON files in [`results/`](results/).

**What the numbers say**

- **Zero-shot is not usable at this model size.** 39.5% of the answers are not one of the 77 labels at all: the
  model invents plausible names (`bank_transfer`, `payment_not_recognised`, `get_virtual_card`) instead of copying
  one from the list. A 0.5B model cannot reliably pick from a 77-item list held in a 450-token prompt.
- **Retrieved examples do most of the prompting work.** Eight nearest-neighbour demonstrations take accuracy from
  19% to 76%, because the correct label is usually among the neighbours and the model mostly has to choose
  between them. They also cut invalid outputs from 39.5% to 6.7%.
- **LoRA adds another 16.6 points while making inference 5.5x faster.** The adapter learns the label set, so it
  needs neither the label list nor the examples in its prompt (62 tokens vs 651). Only 5 of 3,080 outputs were
  not a valid label (for example `copy_of_police_report`).
- **Remaining errors are mostly genuinely ambiguous classes.** The most frequent confusions are
  `card_payment_not_recognised` vs `direct_debit_payment_not_recognised`, `card_arrival` vs
  `card_delivery_estimate`, and `top_up_reverted` vs `top_up_failed`. Per-class F1 is at least 0.95 for 28 of the
  77 intents; the weakest is `transfer_not_received_by_recipient` (0.78).
- **For context:** published Banking77 results with fine-tuned sentence encoders are around 93-94% accuracy
  (Casanueva et al., 2020). A 0.5B generative model with a 1.7% LoRA adapter, trained on a laptop, reaches the
  same range.

---

## Contents

1. [Headline results](#headline-results)
2. [Setup](#setup)
3. [Design decisions](#design-decisions)
4. [Training details](#training-details)
5. [Quick start](#quick-start)
6. [CLI reference](#cli-reference)
7. [Reproducing the results](#reproducing-the-results)
8. [Testing and CI](#testing-and-ci)
9. [Project layout](#project-layout)
10. [Limitations](#limitations)
11. [Next steps](#next-steps)

---

## Setup

| | |
|---|---|
| Base model | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (494M parameters, Apache-2.0) |
| Dataset | [`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) (CC-BY-4.0), 77 intents |
| Splits | train 9,000 / validation 1,003 (stratified 10% of the official train split, seed 42) / test 3,080 (official, untouched until the final runs) |
| Metrics | accuracy, macro-F1 over the 77 classes, invalid-output rate; bootstrap 95% CIs; paired bootstrap for differences |
| Hardware | Apple M4 (10-core GPU), 16 GB unified memory, PyTorch MPS backend, bfloat16 |

The three systems compared:

- **Zero-shot**: base model, system prompt with the instruction and the full list of 77 allowed labels.
- **Few-shot (k = 8)**: same prompt, plus the 8 training examples most similar to the query (TF-IDF cosine)
  inserted as previous chat turns. Retrieved demonstrations are a much stronger baseline than a fixed random set
  when there are 77 classes, because the correct label is often among the neighbours.
- **LoRA**: base model plus a rank-16 LoRA adapter, prompted with the instruction only (no label list); the
  label set is learned during fine-tuning.

## Design decisions

**Generative classification, not a classification head.** The model writes the label as text. This keeps the
chat model's interface, lets zero-shot and fine-tuned runs share one evaluation path, and transfers to extraction
tasks. Outputs are mapped to labels by exact match after light normalisation (case, quotes, spaces vs
underscores); if that fails, the longest label that appears verbatim in the output is used, so the base model is
not penalised for answering `Intent: card_arrival`. Anything else counts as an invalid prediction (always wrong).

**Loss on the answer tokens only.** Each example is rendered through the model's chat template as one
conversation (system, user, assistant) and tokenised in one pass; the prompt positions get label `-100`. The
original version of this repo tokenised inputs and targets separately, which misaligned them, and did not mask
padding. `tests/test_sft.py` checks that supervised positions decode to exactly the label plus the end-of-turn
token and that padding never reaches the loss.

**Logits only where the loss needs them.** A causal LM computes a 151,936-way vocabulary distribution at every
position, but here only the ~5 answer tokens at the end of a ~70-token sequence are supervised. Batches are
left-padded so answers always end at the last position, and the model is asked for the logits of the last
*k* + 1 positions only (`logits_to_keep`). A test checks that this gives the same loss as the full computation.
On MPS this was the difference between a run that peaked at 9 GB at about 2 s per step and one that grew past
12 GB and slowed to 10-60 s per step from swapping; it is also why training uses a short explicit PyTorch loop instead of
`transformers.Trainer`.

**Short prompt after fine-tuning.** The base model needs the 77 labels in its prompt (about 450 tokens). The
fine-tuned model does not, so its prompts are about 8x shorter, which is most of its 5.5x inference speed-up.

**Fair baselines.** Few-shot uses retrieved, not random, examples. Zero-shot and few-shot get the same lenient
output parsing as the fine-tuned model. All runs use the same test examples, so differences are compared with a
paired bootstrap rather than by checking whether two intervals overlap.

## Training details

| | |
|---|---|
| Adapter | LoRA r = 16, alpha = 32, dropout 0.05, on all attention and MLP projections (`q,k,v,o,gate,up,down_proj`) |
| Trainable parameters | **8.80M of 502.8M (1.75%)**; adapter file 34 MB |
| Optimisation | AdamW, learning rate 2e-4, 3% linear warmup then cosine decay, gradient clipping 1.0 |
| Batch | 8 sequences x 2 accumulation steps = 16; mean sequence length 69 tokens |
| Schedule | 2 epochs = 1,126 optimizer steps on 9,000 examples |
| Precision | base weights bfloat16, LoRA weights float32 |
| Validation loss (300 held-out examples) | 0.065 after epoch 1, 0.038 after epoch 2 |
| Training time | **60 min** on the Apple M4 GPU (MPS) |
| Peak GPU memory (MPS driver allocation, includes allocator cache) | 9.1 GB training; 2.2-3.3 GB evaluation |
| Libraries | torch 2.15.0, transformers 5.22.2, peft 0.21.2 |

The number of epochs and the learning rate were set before the final run and not tuned on the test set; the
validation loss was still falling at the end of epoch 2, so the adapter is probably slightly under-trained rather
than over-fitted.

QLoRA (4-bit NF4 base weights, `--qlora`) is implemented but was not run for these results: bitsandbytes 4-bit
kernels need a CUDA GPU and the numbers here come from an Apple-silicon Mac. For a 0.5B model the bf16 weights
are only about 1 GB, so QLoRA's memory saving matters much less than it would for a 7B model.

## Quick start

```bash
git clone https://github.com/luqman304/lora-finetuning && cd lora-finetuning
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/lora-ft prepare            # downloads Banking77 (about 1 MB), writes data/processed/
.venv/bin/lora-ft train              # about 60 min on an M4; writes outputs/lora/adapter
.venv/bin/lora-ft predict --adapter outputs/lora/adapter "I still haven't received my new card"
```

```json
{"text": "I still haven't received my new card", "intent": "card_arrival", "raw": "card_arrival"}
{"text": "Can I top up with a cheque?", "intent": "top_up_by_cash_or_cheque", "raw": "top_up_by_cash_or_cheque"}
{"text": "Someone used my card in another country, I don't recognise that payment", "intent": "card_payment_not_recognised", "raw": "card_payment_not_recognised"}
```

Inference in Docker (CPU), with the adapter mounted from the host:

```bash
docker build -t lora-ft .
docker run --rm -v "$PWD/outputs/lora/adapter:/adapter" -v hf-cache:/models lora-ft \
  --adapter /adapter "Can I top up with a cheque?"
```

## CLI reference

| Command | What it does |
|---|---|
| `lora-ft prepare` | Download Banking77, write stratified train/val/test JSONL and `labels.json` |
| `lora-ft train` | Fine-tune a LoRA adapter (`--qlora` for 4-bit on CUDA); options for rank, alpha, epochs, learning rate, subset size |
| `lora-ft eval --name NAME` | Evaluate the base model (`--prompt-style full`, `--shots k`) or an adapter (`--adapter DIR --prompt-style short`); writes `results/NAME.json` and per-example predictions |
| `lora-ft report --names a,b,c --baseline b` | Markdown results table with paired bootstrap differences |
| `lora-ft predict TEXT...` | Classify messages with the base model or an adapter (local path or Hub repo id) |
| `lora-ft merge --adapter DIR` | Merge the adapter into the base weights and save a standalone model |
| `lora-ft push --repo-id USER/NAME` | Upload the adapter with a generated model card to the Hugging Face Hub (asks for confirmation; private by default) |

## Reproducing the results

```bash
scripts/run_all.sh
```

This trains the adapter, evaluates all three systems on the test split and writes `results/RESULTS.md`. Seeds
are fixed (data split, shuffling, LoRA initialisation). Total time on the M4 was about 1 h 40 min (training 60 min, the three evaluations 40 min). On a CUDA GPU
the same commands run unchanged; add `--qlora` to `train` for 4-bit base weights.

`results/` contains the metrics JSON for every run and `results/train_summary.json` with the full training log,
configuration, parameter counts, timing and library versions. Per-example predictions are written to
`results/*.predictions.jsonl` (not committed).

## Testing and CI

```bash
make test   # 20 offline tests, about 5 s
make lint   # ruff check + ruff format --check
```

The tests build a tiny random Qwen2-architecture model and a byte-level BPE tokenizer on the fly, so they run
offline on CPU and need no model download. They cover data splitting, prompt construction and output parsing,
label masking and padding, equivalence of the answer-only loss with the full loss, the metrics against
hand-computed values, and an end-to-end run: train, save, reload, evaluate, report, merge (merged weights must
reproduce the adapter's logits).

GitHub Actions runs lint and tests on every push and builds the Docker image. No training runs in CI.

## Project layout

```
src/lora_ft/
  cli.py        typer CLI: prepare, train, eval, report, predict, merge, push
  data.py       Banking77 download, label normalisation, stratified split, JSONL I/O
  prompts.py    system prompts, few-shot chat turns, output parsing
  fewshot.py    TF-IDF nearest-neighbour demonstration retrieval (numpy)
  sft.py        chat-template encoding with answer-only labels, padding collator, answer-only loss
  modeling.py   model loading (bf16 / 4-bit), LoRA setup, adapter loading and merging, memory monitor
  train.py      training loop (AdamW, warmup + cosine, gradient accumulation, validation loss)
  evaluate.py   batched greedy generation and scoring
  metrics.py    accuracy, macro-F1, bootstrap and paired-bootstrap confidence intervals
  report.py     results table
  hub.py        Hugging Face Hub upload with model card
tests/          offline pytest suite (tiny random model)
scripts/        run_all.sh, end-to-end reproduction
results/        committed metrics and training summary
```

## Limitations

- **One run per system.** The confidence intervals cover test-set sampling, not training randomness. A few
  training seeds would show how much the LoRA number moves between runs.
- **No hyperparameter search.** Rank, learning rate and epochs are common defaults; the 300-example validation
  subset was used only to monitor loss. A small sweep might add a point or two.
- **QLoRA not measured.** It needs CUDA. The code path exists (`--qlora`) and is the natural next run on a T4.
- **Memory figures on MPS are driver allocations**, which include PyTorch's allocator cache, so they overstate
  what the model strictly needs. CUDA runs report `torch.cuda.max_memory_allocated` instead.
- **Lenient parsing helps the baselines a little.** Base-model outputs that contain a valid label inside other
  text are accepted; with strict exact match their scores would be slightly lower.
- **Single dataset and language.** Banking77 is English, single-domain and clean; real support traffic has
  typos, several intents per message and out-of-scope requests, none of which are modelled here.

## Next steps

- Run QLoRA and a 1.5B / 3B model on a CUDA GPU to measure how the gap to prompting changes with model size.
- Sweep LoRA rank and training-set size (for example 10, 50, 100% of the data) to find how many labelled
  examples are needed to beat few-shot prompting.
- Apply the same pipeline to a structured extraction task (JSON output, entity-level F1).

---

Banking77: Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*, 2020, CC-BY-4.0.
Qwen2.5: Apache-2.0.
