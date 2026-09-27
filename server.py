"""Darwin-27B-JEV: a /v1/systemone decision server.

Fast path: AutoJev-27B answers every typed question (batched across requests).
Reasoning path: a self-contained multiple-choice question on which the fast path
is not confident is handed to Darwin-27B-RSI (any OpenAI-compatible endpoint),
and its answer is returned.

The hand-off rule looks only at the input structure and the fast path's
confidence, never at benchmark identity.

Environment:
  AUTOJEV_CHECKPOINT  path to denis-pplx/autojev-27b          (required)
  REASONER_URLS       comma-separated chat-completions URLs   (required unless TAU=0)
  REASONER_MODEL      served model name                       (default: darwin-27b-rsi)
  TAU                 confidence threshold                    (default: 0.6)
  MAX_OPTIONS         max options for a hand-off              (default: 10)
  PORT                                                         (default: 8090)
Requires the autojev package (github.com/denis-pplx/autojev) on PYTHONPATH.
"""
import asyncio, itertools, json, os, queue, re, threading, time
from concurrent.futures import Future

import httpx, torch
from fastapi import FastAPI, HTTPException
from autojev.model import DecisionModel, answer
from autojev.server import EvaluationRequest

TAU = float(os.getenv("TAU", "0.6"))
MAX_OPTIONS = int(os.getenv("MAX_OPTIONS", "10"))
REASONER_URLS = [u for u in os.getenv("REASONER_URLS", "").split(",") if u]
REASONER_MODEL = os.getenv("REASONER_MODEL", "darwin-27b-rsi")
MAX_LEN = int(os.getenv("MAX_LEN", "32768"))
BATCH = int(os.getenv("BATCH", "8"))
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "24000"))
PROBLEM_KEYS = {"question", "code", "input", "task"}

model = DecisionModel(checkpoint=os.environ["AUTOJEV_CHECKPOINT"])
work: "queue.Queue[tuple[dict, Future]]" = queue.Queue()
urls = itertools.cycle(REASONER_URLS or [""])
stats = {"requests": 0, "questions": 0, "handed_off": 0, "reasoner_used": 0, "reasoner_failed": 0}


def self_contained(state) -> bool:
    """True when the state carries only the problem itself (no documents, catalogs, dialogues...)."""
    if state in ("", None, {}, []) or isinstance(state, str):
        return True
    return isinstance(state, dict) and set(state) <= PROBLEM_KEYS and bool(set(state) & {"question", "code"})


def length(row) -> int:
    return len(json.dumps([row["state"], row["question"]], ensure_ascii=False)) // 3 + 192


def forward(rows):
    with torch.inference_mode():
        batch = model.prepare(rows, max_length=MAX_LEN)
        probs = (model(batch) / model.temperature).softmax(-1).cpu().tolist()
    return [p[:n] for p, n in zip(probs, batch.counts)]


def run_single(row, fut):
    for attempt in range(2):
        try:
            fut.set_result(forward([row])[0]); return
        except torch.OutOfMemoryError:
            pass
        except ValueError as e:
            fut.set_exception(ValueError(str(e))); return
        except Exception as e:
            fut.set_exception(RuntimeError(str(e)[:300])); return
        import gc; gc.collect(); torch.cuda.empty_cache()
    fut.set_exception(RuntimeError("CUDA out of memory"))


def worker():
    while True:
        items = [work.get()]
        t0 = time.time()
        while len(items) < BATCH * 4 and time.time() - t0 < 0.02:
            try: items.append(work.get(timeout=0.005))
            except queue.Empty: pass
        items.sort(key=lambda x: length(x[0]))
        i = 0
        while i < len(items):
            group, tokens = [items[i]], length(items[i][0])
            while i + len(group) < len(items) and len(group) < BATCH and tokens + length(items[i + len(group)][0]) <= TOKEN_BUDGET:
                tokens += length(items[i + len(group)][0]); group.append(items[i + len(group)])
            i += len(group)
            try:
                for (row, fut), p in zip(group, forward([g[0] for g in group])):
                    fut.set_result(p)
            except Exception:
                import gc; gc.collect(); torch.cuda.empty_cache()
                for row, fut in group:
                    if not fut.done(): run_single(row, fut)


threading.Thread(target=worker, daemon=True).start()


def text(x): return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


async def reason(client, state, q):
    prompt = ("State:\n" + (text(state) if state not in ("", None, {}, []) else "(empty)")
              + "\n\nQuestion:\n" + text(q.get("instructions", ""))
              + "\n\nOptions:\n" + "\n".join(f"{k}: {text(v) if v is not None else k}" for k, v in q["criteria"].items())
              + "\n\nThink carefully, then on the final line write exactly: ANSWER: <option key>")
    body = {"model": REASONER_MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6, "top_p": 0.95, "max_tokens": 16000,
            "chat_template_kwargs": {"enable_thinking": True}}
    for _ in range(2):
        try:
            r = await client.post(next(urls), json=body, timeout=1800)
            content = (r.json()["choices"][0]["message"].get("content") or "").split("</think>")[-1]
            found = re.findall(r"ANSWER:\s*\**\s*([^\s*\n]+)", content)
            key = found[-1].strip(" .`'\"") if found else None
            if key in q["criteria"]: return key
        except Exception:
            await asyncio.sleep(2)
    return None


app = FastAPI(title="Darwin-27B-JEV")
client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup():
    global client
    client = httpx.AsyncClient(limits=httpx.Limits(max_connections=256))


@app.get("/health")
def health():
    return {"status": "ready", "model": "darwin-27b-jev", "tau": TAU, "max_options": MAX_OPTIONS, "stats": stats}


@app.post("/v1/systemone")
async def system_one(body: EvaluationRequest):
    stats["requests"] += 1
    questions = {k: q.model_dump(exclude_none=True) for k, q in body.questions.items()}
    images = list(body.images)
    futures = {}
    for k, q in questions.items():
        f = Future(); work.put(({"state": body.state, "question": q, "images": images}, f)); futures[k] = f
    try:
        dists = {k: await asyncio.wrap_future(f) for k, f in futures.items()}
    except ValueError as e:
        raise HTTPException(422, str(e))
    answers, hand_off = {}, []
    for k, q in questions.items():
        try: a = answer(q, dists[k])
        except ValueError: a = answer(q, [1.0] * len(dists[k]))
        answers[k] = a; stats["questions"] += 1
        if (TAU > 0 and q["type"] == "choice" and len(q["criteria"]) <= MAX_OPTIONS and not images
                and self_contained(body.state) and max(a["probabilities"].values()) < TAU):
            hand_off.append(k)
    if hand_off:
        stats["handed_off"] += len(hand_off)
        keys = await asyncio.gather(*[reason(client, body.state, questions[k]) for k in hand_off])
        for k, key in zip(hand_off, keys):
            if key is None: stats["reasoner_failed"] += 1; continue
            stats["reasoner_used"] += 1
            p = answers[k]["probabilities"]; rest = sum(v for kk, v in p.items() if kk != key) or 1.0
            new = {kk: (0.9 if kk == key else 0.1 * v / rest) for kk, v in p.items()}
            n = len(new)
            answers[k] = {**answers[k], "choice": key, "probabilities": new,
                          "confidence": 1.0 if n == 1 else (0.9 - 1 / n) / (1 - 1 / n)}
    return {"model": "darwin-27b-jev", "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8090")), log_level="warning")
