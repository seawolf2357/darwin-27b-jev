# Darwin-27B-JEV engine

Decision server behind the **Darwin-27B-JEV** entry on the [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index).
Results: [FINAL-Bench/Darwin-27B-JEV-decision-index](https://huggingface.co/datasets/FINAL-Bench/Darwin-27B-JEV-decision-index).

## How it decides

1. **Fast path** — AutoJev-27B scores every typed question in one forward pass (requests are batched).
2. **Reasoning path** — if the question is multiple choice (≤ 10 options), its state carries only the problem itself, and the fast path's top probability is below `TAU` (0.6), the question goes to [Darwin-27B-RSI](https://huggingface.co/FINAL-Bench/Darwin-27B-RSI) in thinking mode; its `ANSWER:` replaces the choice (0.9 on the chosen option). If the reasoner fails, the fast-path answer stands.

The rule uses input structure and confidence only, never benchmark identity.

## Run

```bash
# 1. reasoner (any OpenAI-compatible server)
vllm serve FINAL-Bench/Darwin-27B-RSI --served-model-name darwin-27b-rsi --max-model-len 32768 --port 8100

# 2. decision server
git clone https://github.com/denis-pplx/autojev
hf download denis-pplx/autojev-27b --local-dir autojev-27b
AUTOJEV_CHECKPOINT=autojev-27b PYTHONPATH=autojev/src \
REASONER_URLS=http://127.0.0.1:8100/v1/chat/completions \
python server.py            # listens on :8090

# 3. Decision Index kit
python -m decision_index run --engine http --option base_url=http://127.0.0.1:8090 \
  --option model=autojev --rows <rows.jsonl> --out runs/darwin-27b-jev
```

`TAU=0` disables the reasoning path (fast path only).

### One 96 GB GPU (e.g. RTX PRO 6000)

AutoJev-27B (bf16, ~54 GB) and the Q4_K_M reasoner ([FINAL-Bench/Darwin-27B-RSI-GGUF](https://huggingface.co/FINAL-Bench/Darwin-27B-RSI-GGUF), 16.8 GB) fit together on one GPU (~77 GB measured):

```bash
llama-server -m Darwin-27B-RSI-Q4_K_M.gguf --jinja -ngl 99 -np 4 -c 65536 --port 7931 &
AUTOJEV_CHECKPOINT=autojev-27b PYTHONPATH=autojev/src REASONER_URLS=http://127.0.0.1:7931/v1/chat/completions REASONER_MODEL=darwin-27b-rsi python server.py
```

## Latency

Most requests take the fast path (AutoJev-27B alone). Only self-contained multiple-choice questions on which the fast path is unsure go to the reasoner, so the median request is a fast-path request.

## Notes

- Our full run was executed in two passes (fast path on all questions, then the reasoning path on hand-offs). The rule is deterministic, so this matches running this server in one pass.
- Sampling for the reasoner: temperature 0.6, top_p 0.95, max 16,000 tokens.

License: Apache-2.0. Not affiliated with TypeSafe AI.
