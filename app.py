# app.py
# Real-time ISL sign recognition with Ollama-powered sentence formation
#
# HOW TO USE:
#   1. Stand in front of webcam with good lighting
#   2. Press SPACE to start recording a sign
#   3. Perform the sign clearly — prediction appears automatically
#   4. Keep signing — sentence forms automatically after 3s inactivity
#   5. Press C to manually trigger sentence formation
#   6. Press S to speak the formed sentence
#   7. Press R to reset everything
#   8. Press Q to quit

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import threading
import time
from tensorflow.keras.models import load_model  # type: ignore
from data_collection import actions, sequence_length
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic
import sentence_formation as sf

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = load_model('action.h5')
print(f"Model ready. {len(actions)} actions loaded.")

# ── Warm up Ollama in background while app starts ─────────────────────────────
sf.warmup_ollama()

# ── Settings ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.80
COOLDOWN             = 2.0    # seconds before same sign can repeat

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

# ── Predict from sequence ─────────────────────────────────────────────────────
def predict_sign(sequence):
    input_data = np.expand_dims(sequence, axis=0)
    res        = model.predict(input_data, verbose=0)[0]
    idx        = np.argmax(res)
    return actions[idx], float(res[idx])

# ── Draw UI ───────────────────────────────────────────────────────────────────
def draw_ui(image, state, countdown_val, current_sign, confidence,
            recording, inactivity_progress):
    h, w = image.shape[:2]

    buf             = sf.get_buffer()
    formed_sentence = sf.get_sentence()
    ollama_status   = sf.get_status()

    # ── Row 1 — sign buffer ───────────────────────────────────────────────────
    cv2.rectangle(image, (0, 0), (w, 48), (20, 20, 20), -1)
    cv2.putText(image, "Signs:", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1)
    buffer_text = '  >  '.join(buf[-6:]) if buf else '---'
    cv2.putText(image, buffer_text, (65, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    # ── Row 2 — formed sentence ───────────────────────────────────────────────
    cv2.rectangle(image, (0, 48), (w, 96), (12, 30, 12), -1)
    cv2.putText(image, "Sentence:", (8, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 150, 70), 1)

    if ollama_status == "thinking":
        dots = "." * (int(time.time() * 2) % 4)
        cv2.putText(image, f"Forming{dots}", (90, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1)
    elif formed_sentence:
        display = (formed_sentence if len(formed_sentence) < 55
                   else formed_sentence[:52] + "...")
        cv2.putText(image, display, (90, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)
    else:
        cv2.putText(image, '---', (90, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1)

    # Ollama ready indicator (top right)
    from sentence_formation import _ollama_ready
    dot_color   = (0, 255, 0) if _ollama_ready else (0, 100, 255)
    status_text = "AI ready" if _ollama_ready else "AI loading"
    cv2.circle(image, (w - 12, 12), 6, dot_color, -1)
    cv2.putText(image, status_text, (w - 80, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, dot_color, 1)

    # ── Inactivity progress bar ───────────────────────────────────────────────
    if buf and inactivity_progress > 0 and ollama_status == "idle":
        bar_w = int((w - 20) * inactivity_progress)
        cv2.rectangle(image, (10, 91), (10 + bar_w, 95), (0, 180, 255), -1)

    # ── Recording indicator ───────────────────────────────────────────────────
    if recording:
        cv2.circle(image, (w - 15, 70), 8, (0, 0, 255), -1)

    # ── Middle — state ────────────────────────────────────────────────────────
    mid_y = int(h / 2) + 30
    if state == "idle":
        cv2.putText(image, "Press SPACE to sign",
                    (int(w/2) - 130, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    elif state == "countdown":
        cv2.putText(image, f"Get ready...  {countdown_val}",
                    (int(w/2) - 110, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    elif state == "recording":
        cv2.putText(image, "SIGNING NOW...",
                    (int(w/2) - 120, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    elif state == "result":
        color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 140, 255)
        tag   = "" if confidence >= CONFIDENCE_THRESHOLD else " (low)"
        cv2.putText(image, f"{current_sign}{tag}  {confidence:.0%}",
                    (int(w/2) - 140, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

    # ── Bottom bar ────────────────────────────────────────────────────────────
    cv2.rectangle(image, (0, h - 36), (w, h), (20, 20, 20), -1)
    cv2.putText(image,
                "SPACE=sign   C=form sentence   S=speak   R=reset   Q=quit",
                (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1)

    return image

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_sign      = ""
    last_sign_time = 0.0
    current_sign   = ""
    confidence     = 0.0

    state           = "idle"
    sequence        = []
    countdown_val   = 3
    countdown_start = 0.0
    recording       = False

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        print("\nApp started. Press SPACE to record a sign.")
        print("Sentence forms automatically after 3s, or press C.\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame          = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            now = time.time()

            # ── Inactivity tick ───────────────────────────────────────────────
            inactivity_progress = sf.tick()

            # ── Pick up completed Ollama result ───────────────────────────────
            sf.consume_result()

            # ── State machine ─────────────────────────────────────────────────
            if state == "countdown":
                elapsed   = now - countdown_start
                remaining = 3 - int(elapsed)
                if remaining <= 0:
                    state     = "recording"
                    sequence  = []
                    recording = True
                else:
                    countdown_val = remaining

            elif state == "recording":
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)

                if len(sequence) >= sequence_length:
                    recording    = False
                    label, conf  = predict_sign(sequence[-sequence_length:])
                    current_sign = label
                    confidence   = conf
                    state        = "result"

                    if conf >= CONFIDENCE_THRESHOLD:
                        cooldown_ok = (label != last_sign or
                                       now - last_sign_time > COOLDOWN)
                        if cooldown_ok:
                            added = sf.add_sign(label)
                            if added:
                                last_sign      = label
                                last_sign_time = now
                                speak(label)
                                print(f"Added: {label} ({conf:.0%})  "
                                      f"Buffer: {sf.get_buffer()}")
                    else:
                        print(f"Low confidence: {label} ({conf:.0%}) — skipped")

            # ── Draw ──────────────────────────────────────────────────────────
            image = draw_ui(image, state, countdown_val, current_sign,
                            confidence, recording, inactivity_progress)
            cv2.imshow("ISL Recognition", image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):
                if state in ("idle", "result"):
                    state           = "countdown"
                    countdown_start = now
                    countdown_val   = 3
                    current_sign    = ""
                    confidence      = 0.0
                elif state == "recording":
                    recording = False
                    if len(sequence) >= 10:
                        while len(sequence) < sequence_length:
                            sequence.append(sequence[-1])
                        label, conf  = predict_sign(sequence[-sequence_length:])
                        current_sign = label
                        confidence   = conf
                        state        = "result"
                        if conf >= CONFIDENCE_THRESHOLD:
                            cooldown_ok = (label != last_sign or
                                           now - last_sign_time > COOLDOWN)
                            if cooldown_ok:
                                added = sf.add_sign(label)
                                if added:
                                    last_sign      = label
                                    last_sign_time = now
                                    speak(label)
                    else:
                        state = "idle"

            elif key == ord('c'):
                if sf.get_buffer() and sf.get_status() != "thinking":
                    print(f"Manual C trigger: {sf.get_buffer()}")
                    sf.form_sentence()

            elif key == ord('s'):
                text = sf.get_sentence() or " ".join(sf.get_buffer())
                if text:
                    speak(text)
                    print(f"Speaking: {text}")

            elif key == ord('r'):
                sf.reset()
                current_sign = ""
                confidence   = 0.0
                last_sign    = ""
                state        = "idle"
                sequence     = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()