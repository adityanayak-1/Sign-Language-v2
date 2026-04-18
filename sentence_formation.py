# sentence_formation.py
# Handles all sentence formation logic — sign buffer, deduplication,
# inactivity detection, and Ollama-based grammar reconstruction.
# Designed to be imported by app.py — no OpenCV or camera logic here.

import threading
import time

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_MODEL       = 'phi3:mini'
INACTIVITY_TIMEOUT = 3.0   # seconds of no new sign → auto form sentence
MAX_BUFFER_SIZE    = 10    # max signs to keep in buffer

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an expert ISL (Indian Sign Language) to English translator.
ISL grammar follows Subject-Object-Verb order, unlike English Subject-Verb-Object.

Your job: receive a comma-separated list of ISL signs and output ONE fluent, \
grammatically correct English sentence.

Strict rules:
1. Reorder words into natural English (Subject-Verb-Object).
2. Add necessary helper words (am, is, are, do, can, need, the, a, my, your...).
3. If the meaning is a question, end with ?  Otherwise end with .
4. Output the sentence ONLY — no explanation, no parentheses, no extra text.
5. Never add words not implied by the signs.
"""

# Few-shot examples — model sees these before every real query.
_FEW_SHOT = [
    {"role": "user",      "content": "Signs: I, Happy"},
    {"role": "assistant", "content": "I am happy."},

    {"role": "user",      "content": "Signs: My, Name, What"},
    {"role": "assistant", "content": "What is my name?"},

    {"role": "user",      "content": "Signs: You, Where, Home"},
    {"role": "assistant", "content": "Where is your home?"},

    {"role": "user",      "content": "Signs: Hello, How, You"},
    {"role": "assistant", "content": "Hello! How are you?"},

    {"role": "user",      "content": "Signs: I, Help"},
    {"role": "assistant", "content": "I need help."},

    {"role": "user",      "content": "Signs: You, Happy, Today, Good"},
    {"role": "assistant", "content": "You are happy and doing well today."},

    {"role": "user",      "content": "Signs: I, Food, Want"},
    {"role": "assistant", "content": "I want food."},

    {"role": "user",      "content": "Signs: I, Work, Today, Finished"},
    {"role": "assistant", "content": "I finished work today."},

    {"role": "user",      "content": "Signs: ThankYou"},
    {"role": "assistant", "content": "Thank you!"},

    {"role": "user",      "content": "Signs: I, Water"},
    {"role": "assistant", "content": "I need water."},

    {"role": "user",      "content": "Signs: You, Name, What"},
    {"role": "assistant", "content": "What is your name?"},

    {"role": "user",      "content": "Signs: I, Sick"},
    {"role": "assistant", "content": "I am sick."},

    {"role": "user",      "content": "Signs: You, Doctor, Go"},
    {"role": "assistant", "content": "You should go to the doctor."},

    {"role": "user",      "content": "Signs: I, School, Go"},
    {"role": "assistant", "content": "I am going to school."},

    {"role": "user",      "content": "Signs: Please, Help, Me"},
    {"role": "assistant", "content": "Please help me."},
]

# ── State ─────────────────────────────────────────────────────────────────────
_state = {
    "sign_buffer":          [],
    "formed_sentence":      "",
    "status":               "idle",   # idle | thinking | done | error
    "last_sign_time":       0.0,
    "inactivity_triggered": False,
    "generation":           0,        # incremented on every form_sentence() call
}
_lock  = threading.Lock()
_event = threading.Event()           # fired when a result is ready

# ── Ollama warm-up ────────────────────────────────────────────────────────────
_ollama_ready = False

def warmup_ollama():
    """Warm up Ollama in background so first real call is fast."""
    def _run():
        global _ollama_ready
        try:
            import ollama
            print("[Ollama] Warming up model...")
            ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": "Signs: Hello"},
                ],
                options={'temperature': 0.1, 'num_predict': 8},
            )
            _ollama_ready = True
            print("[Ollama] Model warm and ready.")
        except Exception as e:
            print(f"[Ollama] Warmup failed: {e}")
            _ollama_ready = False

    threading.Thread(target=_run, daemon=True).start()


# ── Sign buffer ───────────────────────────────────────────────────────────────
def add_sign(sign):
    """Add a sign to the buffer. Returns True if added, False if rejected."""
    if not sign:
        return False
    with _lock:
        buf = _state["sign_buffer"]
        if buf and buf[-1] == sign:          # deduplication
            return False
        if len(buf) >= MAX_BUFFER_SIZE:
            buf.pop(0)
        buf.append(sign)
        _state["last_sign_time"]       = time.time()
        _state["inactivity_triggered"] = False
        return True


def get_buffer():
    with _lock:
        return list(_state["sign_buffer"])

def get_sentence():
    with _lock:
        return _state["formed_sentence"]

def get_status():
    with _lock:
        return _state["status"]


def reset():
    """
    Clear buffer and sentence.
    Increments generation counter so any in-flight Ollama thread
    knows its result is stale and should be discarded.
    """
    with _lock:
        _state["sign_buffer"]          = []
        _state["formed_sentence"]      = ""
        _state["status"]               = "idle"
        _state["last_sign_time"]       = 0.0
        _state["inactivity_triggered"] = False
        _state["generation"]          += 1   # invalidate in-flight threads
    _event.clear()
    print("[Sentence] Reset.")


# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_messages(signs):
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *_FEW_SHOT,
        {"role": "user", "content": f"Signs: {', '.join(signs)}"},
    ]


def _clean(raw):
    """Take only the first line and strip noise."""
    line = raw.split('\n')[0].strip()
    # Strip parenthetical asides like "(Assuming...)"
    if '(' in line:
        line = line[:line.index('(')].strip()
    for prefix in ("Sentence:", "Output:", "ISL:", "English:",
                   "Answer:", "Translation:", "Result:"):
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix):].strip()
    if line and line[-1] not in '.!?':
        line += '.'
    return line


# ── Sentence formation ────────────────────────────────────────────────────────
def form_sentence(signs=None):
    """
    Fire off Ollama in a background thread.
    Uses a generation counter so reset() mid-flight won't corrupt results.
    """
    if signs is None:
        signs = get_buffer()
    if not signs:
        return

    with _lock:
        if _state["status"] == "thinking":
            return
        _state["status"]               = "thinking"
        _state["formed_sentence"]      = ""
        _state["inactivity_triggered"] = True
        my_gen = _state["generation"]   # this thread owns this generation
    _event.clear()

    def _run():
        try:
            import ollama
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=_build_messages(signs),
                options={
                    'temperature':    0.1,
                    'num_predict':    60,
                    'top_p':          0.95,
                    'repeat_penalty': 1.1,
                },
            )
            raw      = response['message']['content'].strip()
            sentence = _clean(raw)

            with _lock:
                # Only commit if we are still the current generation
                if _state["generation"] == my_gen:
                    _state["formed_sentence"] = sentence
                    _state["status"]          = "done"
                    _event.set()

            print(f"[Ollama] → {sentence}")

        except Exception as e:
            print(f"[Ollama] Error: {e}")
            fallback = " ".join(signs).capitalize() + "."
            with _lock:
                if _state["generation"] == my_gen:
                    _state["formed_sentence"] = fallback
                    _state["status"]          = "error"
                    _event.set()

    threading.Thread(target=_run, daemon=True).start()


def wait_for_result(timeout=30):
    """
    Block until the current form_sentence() call completes.
    Returns the sentence string, or '' on timeout.
    Called by test_sentence.py — NOT used by app.py.
    """
    _event.wait(timeout=timeout)
    return get_sentence()


# ── Inactivity auto-trigger ───────────────────────────────────────────────────
def tick():
    """Call once per frame from app.py. Returns inactivity progress 0.0-1.0."""
    with _lock:
        buf       = _state["sign_buffer"]
        last_time = _state["last_sign_time"]
        triggered = _state["inactivity_triggered"]
        status    = _state["status"]

    if not buf or last_time == 0.0 or triggered or status == "thinking":
        return 0.0

    elapsed  = time.time() - last_time
    progress = min(elapsed / INACTIVITY_TIMEOUT, 1.0)

    if elapsed >= INACTIVITY_TIMEOUT:
        print(f"[Sentence] Auto-forming from inactivity: {buf}")
        form_sentence(list(buf))

    return progress


# ── Consume result (used by app.py main loop) ─────────────────────────────────
def consume_result():
    """If a result is ready, return it and reset status to idle."""
    with _lock:
        if _state["status"] in ("done", "error"):
            sentence = _state["formed_sentence"]
            _state["status"] = "idle"
            return sentence
    return None