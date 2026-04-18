# test_sentence.py
# Tests the complete sentence formation pipeline WITHOUT needing camera or model.
# Run with: python test_sentence.py

import time
import threading
import sentence_formation as sf

# ── Wait for Ollama warmup ────────────────────────────────────────────────────
sf.warmup_ollama()
print("Waiting for Ollama warmup...")
for _ in range(60):
    time.sleep(1)
    if sf._ollama_ready:
        break
if not sf._ollama_ready:
    print("WARNING: Ollama did not warm up in time — results may be slow.")
print()

# ── TTS ───────────────────────────────────────────────────────────────────────
def speak(text):
    if not text.strip():
        return
    def _run():
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()


# ── Helper: build deduped buffer ──────────────────────────────────────────────
def build_buffer(signs):
    buf = []
    for sign in signs:
        if not buf or buf[-1] != sign:
            buf.append(sign)
    return buf


# ── Helper: run one sentence formation and block until done ───────────────────
def run_one(signs, timeout=30):
    """
    Resets state, loads signs, fires form_sentence(), then blocks via
    wait_for_result(). Returns (sentence, elapsed_seconds).

    wait_for_result() uses threading.Event — it only unblocks when the
    thread for THIS generation sets the event. No off-by-one results.
    """
    sf.reset()                       # bumps generation counter
    for sign in signs:
        sf.add_sign(sign)            # dedup handled inside add_sign

    start = time.time()
    sf.form_sentence()               # fires background thread
    sentence = sf.wait_for_result(timeout=timeout)   # blocks on Event
    elapsed  = time.time() - start
    return sentence, elapsed


# ── Test cases ────────────────────────────────────────────────────────────────
TEST_CASES = [
    ("Basic statement",               ["I", "Happy"]),
    ("ISL SOV order",                 ["My", "Name", "What"]),
    ("With duplicate (should dedup)", ["I", "I", "Water"]),
    ("Question",                      ["You", "Where", "Home"]),
    ("Need help",                     ["I", "Help"]),
    ("Multi word",                    ["I", "Work", "Today", "Finished"]),
    ("Greeting",                      ["Hello", "How", "You"]),
    ("Want food",                     ["I", "Food", "Want"]),
    ("Thank you",                     ["ThankYou"]),
    ("Complex",                       ["You", "Happy", "Today", "Good"]),
]


def run_tests():
    print("=" * 60)
    print("  SENTENCE FORMATION PIPELINE TEST")
    print("=" * 60)
    print()

    # ── Test 1: Deduplication ─────────────────────────────────────────────────
    print("── TEST 1: Deduplication ────────────────────────────────")
    test_signs = ["Hello", "Hello", "I", "I", "Happy", "Happy", "Happy"]
    print(f"Input (with dupes): {test_signs}")
    buf = build_buffer(test_signs)
    print(f"After dedup:        {buf}")
    assert buf == ["Hello", "I", "Happy"], f"FAIL: got {buf}"
    print("PASS ✓\n")

    # ── Test 2: Sentence formation ────────────────────────────────────────────
    print("── TEST 2: Ollama sentence formation ────────────────────")
    print("Each case blocks until its OWN result arrives (no off-by-one).\n")

    results = []
    for desc, signs in TEST_CASES:
        buf = build_buffer(signs)
        print(f"  Case    : {desc}")
        print(f"  Signs   : {signs}")
        print(f"  Buffer  : {buf}")

        sentence, elapsed = run_one(signs)

        print(f"  Output  : {sentence}")
        print(f"  Latency : {elapsed:.1f}s")
        print()
        results.append((desc, buf, sentence, elapsed))

    # ── Test 3: TTS ───────────────────────────────────────────────────────────
    print("── TEST 3: TTS ──────────────────────────────────────────")
    sample = results[0][2] if results else "Hello, how are you?"
    print(f"Speaking: '{sample}'")
    speak(sample)
    time.sleep(4)
    print("TTS done ✓\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for desc, buf, sentence, elapsed in results:
        print(f"  {desc}")
        print(f"    {buf} → '{sentence}'  ({elapsed:.1f}s)")
        print()

    avg = sum(r[3] for r in results) / len(results)
    print(f"Average latency: {avg:.1f}s")
    print("\nAll tests complete.")


if __name__ == "__main__":
    run_tests()