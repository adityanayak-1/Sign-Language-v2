# data_collection.py
import cv2
import numpy as np
import os
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("MP_DATA")    # keypoints saved here
VIDEOS_PATH = os.path.join("MP_VIDEOS")  # sample videos saved here

# ── 25 signs ──────────────────────────────────────────────────────────────────
actions = np.array([
    "I",
    "You",
    "My",
    "Your",
    "Name",
    "What",
    "Where",
    "When",
    "How",
    "Hello",
    "Food",
    "Water",
    "Home",
    "Work",
    "Help",
    "Happy",
    "Sad",
    "Good",
    "Finished",
    "ThankYou",
    "Today",
    "Monday",
    "Red",
    "Black",
    "White"
])

# Number of videos per sign
no_sequences = 30

# Number of frames per video
sequence_length = 30


# ── Check if a sign is already fully collected ────────────────────────────────
def is_complete(action):
    """Returns True if all 30 videos with 30 frames exist for this sign."""
    for sequence in range(no_sequences):
        for frame_num in range(sequence_length):
            path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            if not os.path.exists(path):
                return False
    return True


# ── Show collection summary ───────────────────────────────────────────────────
def print_summary():
    """Print how many videos are collected for each sign."""
    print("\n" + "="*45)
    print("  COLLECTION SUMMARY")
    print("="*45)
    total_complete = 0
    for action in actions:
        count = 0
        for sequence in range(no_sequences):
            seq_path = os.path.join(DATA_PATH, action, str(sequence))
            if os.path.isdir(seq_path):
                frames = len([f for f in os.listdir(seq_path) if f.endswith(".npy")])
                if frames == sequence_length:
                    count += 1
        status = "✓ COMPLETE" if count == no_sequences else f"{count}/{no_sequences} videos"
        if count == no_sequences:
            total_complete += 1
        print(f"  {action:<15} {status}")
    print("="*45)
    print(f"  Total complete: {total_complete}/{len(actions)} signs")
    print("="*45 + "\n")


# ── Wait for keypress screen ──────────────────────────────────────────────────
def wait_for_keypress(cap, holistic, message, sub_message="Press SPACE to start or Q to quit"):
    """Show message on screen and wait for SPACE to continue."""
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        cv2.putText(image, message, (60, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(image, sub_message, (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord(' '):
            return True
        elif key == ord('q'):
            return False


# ── Collect one sign ──────────────────────────────────────────────────────────
def collect_sign(cap, holistic, action, action_index, h, w):
    """Collect all 30 videos for one sign."""

    # Sample video writer (sequence 0 only)
    sample_path = os.path.join(VIDEOS_PATH, f"{action}.mp4")
    fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
    writer      = None

    for sequence in range(no_sequences):

        # Start video writer for sequence 0
        if sequence == 0:
            writer = cv2.VideoWriter(sample_path, fourcc, 15, (w, h))

        for frame_num in range(sequence_length):

            ret, frame = cap.read()
            if not ret:
                continue

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Progress text — top left
            cv2.putText(image,
                        "Sign {}/{}: {}  |  Video {}/{}".format(
                            action_index, len(actions), action,
                            sequence + 1, no_sequences),
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # First frame of each video — pause so recorder can reset
            if frame_num == 0:
                cv2.putText(image, "GET READY...", (180, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(2000)
            else:
                cv2.putText(image, "RECORDING", (220, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

            # Save keypoints
            keypoints = extract_keypoints(results)
            npy_path  = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            # Save actual frame for sample video (sequence 0 only)
            if sequence == 0 and writer is not None:
                writer.write(image)

            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                if writer is not None:
                    writer.release()
                return False  # signal quit

        # Release writer after sequence 0
        if sequence == 0 and writer is not None:
            writer.release()
            writer = None
            print(f"  Sample video saved: {sample_path}")

    print(f"  ✓ {action} complete — {no_sequences} videos collected")
    return True  # signal continue


# ── Main collection ───────────────────────────────────────────────────────────
def collect_all(skip_complete=True):
    """
    Collect data for all signs.
    skip_complete=True  — skip signs already fully collected (resume mode)
    skip_complete=False — recollect everything from scratch
    """
    cap       = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w      = frame.shape[:2]

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:

        for idx, action in enumerate(actions, start=1):

            # Resume — skip already complete signs
            if skip_complete and is_complete(action):
                print(f"  Skipping {action} — already complete")
                continue

            # Show sign name and wait for SPACE
            message     = "NEXT: {} ({}/{})".format(action, idx, len(actions))
            sub_message = "Press SPACE to start  |  Q to quit"
            should_continue = wait_for_keypress(cap, holistic, message, sub_message)

            if not should_continue:
                print("\nCollection stopped by user.")
                break

            # Collect this sign
            print(f"\nCollecting: {action} ({idx}/{len(actions)})")
            completed = collect_sign(cap, holistic, action, idx, h, w)

            if not completed:
                print("\nCollection stopped by user.")
                break

    cap.release()
    cv2.destroyAllWindows()


# ── Recollect one specific sign ───────────────────────────────────────────────
def recollect_sign(action_name):
    """Redo collection for one specific sign only."""
    if action_name not in actions:
        print(f"Error: '{action_name}' not in actions list.")
        print(f"Available signs: {list(actions)}")
        return

    # Clear existing data for this sign
    import shutil
    sign_path = os.path.join(DATA_PATH, action_name)
    if os.path.exists(sign_path):
        shutil.rmtree(sign_path)
        print(f"Cleared existing data for: {action_name}")

    # Recreate folders
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action_name, str(sequence)), exist_ok=True)

    cap        = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w       = frame.shape[:2]
    idx        = list(actions).index(action_name) + 1

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:

        message = "RECOLLECTING: {}".format(action_name)
        should_continue = wait_for_keypress(cap, holistic, message)

        if should_continue:
            print(f"\nRecollecting: {action_name}")
            collect_sign(cap, holistic, action_name, idx, h, w)

    cap.release()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Create all folders
    for action in actions:
        for sequence in range(no_sequences):
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)
    os.makedirs(VIDEOS_PATH, exist_ok=True)

    # Show current progress before starting
    print_summary()

    # Ask user what to do
    print("Options:")
    print("  1 — Resume collection (skip completed signs)")
    print("  2 — Recollect one specific sign")
    print("  3 — Recollect everything from scratch")
    print("  4 — Just show summary and exit")

    choice = input("\nEnter choice (1/2/3/4): ").strip()

    if choice == "1":
        collect_all(skip_complete=True)
        print_summary()

    elif choice == "2":
        sign = input("Enter sign name exactly as listed (e.g. ThankYou, I, Hello): ").strip()
        recollect_sign(sign)
        print_summary()

    elif choice == "3":
        confirm = input("This will delete ALL existing data. Type YES to confirm: ")
        if confirm == "YES":
            import shutil
            if os.path.exists(DATA_PATH):
                shutil.rmtree(DATA_PATH)
            for action in actions:
                for sequence in range(no_sequences):
                    os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)
            collect_all(skip_complete=False)
            print_summary()
        else:
            print("Cancelled.")

    elif choice == "4":
        pass  # summary already printed above

    else:
        print("Invalid choice. Running resume mode by default.")
        collect_all(skip_complete=True)
        print_summary()
