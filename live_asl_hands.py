


#|------------------------------------------------|
#|   added speak Feature with "s" "d" "c" "e"     |
#|------------------------------------------------|


# live_asl_hands.py
import argparse, json, os, time, collections, textwrap, threading
import numpy as np
import cv2

# Optional TF import only if Keras path is provided
import tensorflow.lite as tf

import mediapipe as mp
mp_hands = mp.solutions.hands

# ---------- TTS (pyttsx3) ----------
try:
    import pyttsx3
    _tts_import_ok = True
except Exception:
    pyttsx3 = None
    _tts_import_ok = False

class TTSWrapper:
    def __init__(self, rate=175, volume=1.0, enabled=True):
        self.enabled = enabled and _tts_import_ok
        self.engine = None
        self.lock = threading.Lock()
        if self.enabled:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", int(rate))
                self.engine.setProperty("volume", float(volume))
            except Exception:
                self.engine = None
                self.enabled = False

    def speak(self, text: str):
        if not self.enabled or not self.engine or not text.strip():
            return
        # Run speaking in a background thread, serialize via a lock
        def _worker():
            with self.lock:
                try:
                    self.engine.stop()         # interrupt any ongoing speech
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception:
                    pass
        threading.Thread(target=_worker, daemon=True).start()



# --------------------------
# Args
# --------------------------
parser = argparse.ArgumentParser(description="Live ASL inference with MediaPipe Hands + text builder + TTS")
parser.add_argument("--keras",   type=str, default=None, help="Path to Keras .keras model (optional)")
parser.add_argument("--tflite",  type=str, default=None, help="Path to TFLite .tflite model (optional)")
parser.add_argument("--classes", type=str, required=True, help="Path to class_indices.json")
parser.add_argument("--cam",     type=int, default=0, help="Webcam index")
parser.add_argument("--size",    type=int, default=224, help="Model input size (default 224)")
parser.add_argument("--smooth",  type=int, default=10, help="Temporal smoothing window")
parser.add_argument("--min_det", type=float, default=0.5, help="MediaPipe min_detection_confidence")
parser.add_argument("--min_track", type=float, default=0.5, help="MediaPipe min_tracking_confidence")
parser.add_argument("--pad",     type=float, default=0.25, help="Padding ratio around hand box")
parser.add_argument("--min_conf",type=float, default=0.0, help="Min confidence to accept char on 's'")
# TTS options
parser.add_argument("--tts",        type=int, default=1,   help="Enable TTS on 'e' (1=yes,0=no)")
parser.add_argument("--tts_rate",   type=int, default=175, help="TTS speaking rate (wpm)")
parser.add_argument("--tts_volume", type=float, default=1.0, help="TTS volume (0.0-1.0)")
args = parser.parse_args()

if not args.keras and not args.tflite:
    raise SystemExit("Provide either --keras or --tflite")

IMAGE_SIZE = args.size

# --------------------------
# Load classes
# --------------------------
with open(args.classes, "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}
num_classes = len(idx_to_class)

# Optional mapping for special labels from ASL dataset
SPECIAL_MAP = {
    "space": " ",
    "nothing": ""  # ignore "nothing"
}

# --------------------------
# Load model (Keras or TFLite)
# --------------------------
use_keras = args.keras is not None
if use_keras:
    if tf is None:
        raise SystemExit("TensorFlow not available; install tensorflow or use --tflite")
    print("Loading Keras model:", args.keras)
    model = tf.keras.models.load_model(args.keras)
else:
    print("Loading TFLite model:", args.tflite)
    import tensorflow as tf  # minimal runtime
    interpreter = tf.lite.Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

# --------------------------
# Helpers
# --------------------------
def preprocess_rgb(rgb):
    img = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    x = img.astype(np.float32) / 255.0
    return np.expand_dims(x, 0)

def predict_probs(batch):
    if use_keras:
        return model.predict(batch, verbose=0)[0]
    else:
        interpreter.set_tensor(in_det[0]['index'], batch)
        interpreter.invoke()
        return interpreter.get_tensor(out_det[0]['index'])[0]

def label_from_probs(p):
    top = int(np.argmax(p))
    return idx_to_class[top], float(p[top])

def xywh_from_landmarks(landmarks, W, H, pad_ratio=0.25):
    xs = [lm.x for lm in landmarks]; ys = [lm.y for lm in landmarks]
    x1 = max(0.0, min(xs)); y1 = max(0.0, min(ys))
    x2 = min(1.0, max(xs)); y2 = min(1.0, max(ys))
    x1a, y1a, x2a, y2a = int(x1*W), int(y1*H), int(x2*W), int(y2*H)
    bw, bh = x2a - x1a, y2a - y1a
    cx, cy = (x1a + x2a) // 2, (y1a + y2a) // 2
    side = int(max(bw, bh) * (1.0 + pad_ratio))
    side = max(32, side)
    x1p, y1p = cx - side//2, cy - side//2
    x2p, y2p = x1p + side, y1p + side
    x1p = max(0, min(W-1, x1p)); y1p = max(0, min(H-1, y1p))
    x2p = max(1, min(W, x2p));   y2p = max(1, min(H, y2p))
    return x1p, y1p, x2p, y2p

def draw_panel(frame, lines, color=(0,0,0), bg=(255,255,255)):
    pad = 8; lh = 22
    w = max([cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for t in lines] + [200]) + pad*2
    h = lh*len(lines) + pad*2
    overlay = frame.copy()
    cv2.rectangle(overlay, (10,10), (10+w, 10+h), bg, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    y = 10 + pad + 16
    for t in lines:
        cv2.putText(frame, t, (10+pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += lh

def draw_textbox(frame, header, text, max_width_ratio=0.9, color=(255,255,255), txt=(0,0,0)):
    H, W = frame.shape[:2]
    max_w = int(W * max_width_ratio)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.7, 2
    pad = 10
    line_h = 24

    # simple width-based wrap
    wrapped = []
    cur = ""
    for ch in text if text else "":
        nxt = cur + ch
        w, _ = cv2.getTextSize(nxt, font, fs, th)[0]
        if w > max_w - 2*pad:
            wrapped.append(cur)
            cur = ch
        else:
            cur = nxt
    if cur or not text:
        wrapped.append(cur)

    lines = [header] + wrapped
    box_w = max([cv2.getTextSize(l if l else " ", font, fs, th)[0][0] for l in lines] + [200]) + 2*pad
    box_h = line_h * len(lines) + 2*pad

    x1 = (W - box_w) // 2
    y1 = H - box_h - 10
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = y1 + pad + 16
    for i, l in enumerate(lines):
        if i == 0:
            cv2.putText(frame, l, (x1+pad, y), font, 0.75, (0,0,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, l if l else " ", (x1+pad, y), font, fs, txt, th, cv2.LINE_AA)
        y += line_h

# --------------------------
# Init TTS
# --------------------------
tts = TTSWrapper(rate=args.tts_rate, volume=args.tts_volume, enabled=bool(args.tts))
if args.tts and not tts.enabled:
    print("TTS requested but pyttsx3 not available/failed to init. Install with: pip install pyttsx3")

# --------------------------
# Loop
# --------------------------
cap = cv2.VideoCapture(args.cam)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {args.cam}")

pred_queue = collections.deque(maxlen=max(1, args.smooth))
fps_clock = collections.deque(maxlen=30)
flip_view = True
show_box  = True

# Text builder state
saved_chars = []

print("Controls: q=quit  f=flip  b=box  s=save-char  d=delete  c=clear  e=speak+print")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=args.min_det,
    min_tracking_confidence=args.min_track
) as hands:

    while True:
        t0 = time.time()
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if flip_view:
            frame_bgr = cv2.flip(frame_bgr, 1)

        H, W = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Detect hand
        res = hands.process(frame_rgb)
        crop_rgb = None
        box = None

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            x1, y1, x2, y2 = xywh_from_landmarks(lm, W, H, pad_ratio=args.pad)
            box = (x1, y1, x2, y2)
            crop_rgb = frame_rgb[y1:y2, x1:x2]

        # Use crop if available, otherwise full-frame
        if crop_rgb is None or crop_rgb.size == 0:
            crop_rgb = frame_rgb
            box = None

        batch = preprocess_rgb(crop_rgb)
        probs = predict_probs(batch)
        pred_queue.append(probs)
        avg_probs = np.mean(np.stack(pred_queue, 0), 0)
        label, conf = label_from_probs(avg_probs)

        # Draw hand box
        if show_box and box is not None:
            cv2.rectangle(frame_bgr, (box[0],box[1]), (box[2],box[3]), (0,255,255), 2)

        # FPS + HUD
        fps_clock.append(1.0 / max(1e-6, (time.time()-t0)))
        fps = sum(fps_clock)/len(fps_clock)
        hud = [
            f"Pred: {label} ({conf:.2f})",
            f"FPS: {fps:.1f}",
            "Keys: q quit | f flip | b box | s save | d del | c clear | e speak"
        ]
        draw_panel(frame_bgr, hud)

        # Saved text box
        text_str = "".join(saved_chars)
        draw_textbox(frame_bgr, header="Saved Text:", text=text_str)

        # Key handling
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('f'):
            flip_view = not flip_view
        elif k == ord('b'):
            show_box = not show_box
        elif k == ord('s'):
            # append current predicted label (respect min_conf)
            to_add = SPECIAL_MAP.get(label, label)
            if conf >= args.min_conf and to_add is not None:
                # add characters one-by-one (most are length 1)
                saved_chars.extend(list(to_add))
        elif k == ord('d'):
            if saved_chars:
                saved_chars.pop()
        elif k == ord('c'):
            saved_chars.clear()
        elif k == ord('e'):
            # speak and print the string
            out = "".join(saved_chars).strip()
            print("Saved string:", out if out else "(empty)")
            if out and args.tts and tts.enabled:
                tts.speak(out)

        cv2.imshow("ASL Live (MediaPipe Hands) + Text Builder + TTS", frame_bgr)

cap.release()
cv2.destroyAllWindows()
try:
    tts.shutdown()
except Exception:
    pass


# live_asl_hands_fixed.py

import argparse, json, os, time, collections, textwrap, threading
import numpy as np
import cv2

# Optional TF import only if Keras path is provided
try:
    import tensorflow as tf
except Exception:
    tf = None

import mediapipe as mp
mp_hands = mp.solutions.hands

# ---------- TTS (pyttsx3) ----------
try:
    import pyttsx3
    _tts_import_ok = True
except Exception:
    pyttsx3 = None
    _tts_import_ok = False

class TTSWrapper:
    def __init__(self, rate=175, volume=1.0, enabled=True):
        self.enabled = enabled and _tts_import_ok
        self.engine = None
        self.lock = threading.Lock()
        if self.enabled:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", int(rate))
                self.engine.setProperty("volume", float(volume))
            except Exception:
                self.engine = None
                self.enabled = False

    def speak(self, text: str):
        if not self.enabled or not self.engine or not text.strip():
            return
        # Run speaking in a background thread, serialize via a lock
        def _worker():
            with self.lock:
                try:
                    self.engine.stop()         # interrupt any ongoing speech
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception:
                    pass
        threading.Thread(target=_worker, daemon=True).start()

    def shutdown(self):
        try:
            if self.engine:
                with self.lock:
                    # pyttsx3 doesn't guarantee a standard shutdown API, but stop is safe
                    self.engine.stop()
        except Exception:
            pass


# --------------------------
# Args
# --------------------------
parser = argparse.ArgumentParser(description="Live ASL inference with MediaPipe Hands + text builder + TTS")
parser.add_argument("--keras",   type=str, default=None, help="Path to Keras .keras model (optional)")
parser.add_argument("--tflite",  type=str, default=None, help="Path to TFLite .tflite model (optional)")
parser.add_argument("--classes", type=str, required=True, help="Path to class_indices.json")
parser.add_argument("--cam",     type=int, default=0, help="Webcam index")
parser.add_argument("--size",    type=int, default=224, help="Model input size (default 224)")
parser.add_argument("--smooth",  type=int, default=10, help="Temporal smoothing window")
parser.add_argument("--min_det", type=float, default=0.5, help="MediaPipe min_detection_confidence")
parser.add_argument("--min_track", type=float, default=0.5, help="MediaPipe min_tracking_confidence")
parser.add_argument("--pad",     type=float, default=0.25, help="Padding ratio around hand box")
parser.add_argument("--min_conf",type=float, default=0.0, help="Min confidence to accept char on 's'")
# TTS options
parser.add_argument("--tts",        type=int, default=1,   help="Enable TTS on 'e' (1=yes,0=no)")
parser.add_argument("--tts_rate",   type=int, default=175, help="TTS speaking rate (wpm)")
parser.add_argument("--tts_volume", type=float, default=1.0, help="TTS volume (0.0-1.0)")
args = parser.parse_args()

if not args.keras and not args.tflite:
    raise SystemExit("Provide either --keras or --tflite")

IMAGE_SIZE = args.size

# --------------------------
# Load classes
# --------------------------
with open(args.classes, "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}
num_classes = len(idx_to_class)

# Optional mapping for special labels from ASL dataset
SPECIAL_MAP = {
    "space": " ",
    "nothing": ""  # ignore "nothing"
}

# --------------------------
# Load model (Keras or TFLite)
# --------------------------
use_keras = args.keras is not None
if use_keras:
    if tf is None:
        raise SystemExit("TensorFlow not available; install tensorflow or use --tflite")
    print("Loading Keras model:", args.keras)
    model = tf.keras.models.load_model(args.keras)
else:
    print("Loading TFLite model:", args.tflite)
    import tensorflow as tf  # minimal runtime
    interpreter = tf.lite.Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

# --------------------------
# Helpers
# --------------------------
def preprocess_rgb(rgb):
    img = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    x = img.astype(np.float32) / 255.0
    return np.expand_dims(x, 0)

def predict_probs(batch):
    # ensure correct dtype
    batch = batch.astype(np.float32)
    if use_keras:
        return model.predict(batch, verbose=0)[0]
    else:
        interpreter.set_tensor(in_det[0]['index'], batch)
        interpreter.invoke()
        return interpreter.get_tensor(out_det[0]['index'])[0]

def label_from_probs(p):
    top = int(np.argmax(p))
    return idx_to_class[top], float(p[top])

def xywh_from_landmarks(landmarks, W, H, pad_ratio=0.25):
    xs = [lm.x for lm in landmarks]; ys = [lm.y for lm in landmarks]
    x1 = max(0.0, min(xs)); y1 = max(0.0, min(ys))
    x2 = min(1.0, max(xs)); y2 = min(1.0, max(ys))
    x1a, y1a, x2a, y2a = int(x1*W), int(y1*H), int(x2*W), int(y2*H)
    bw, bh = x2a - x1a, y2a - y1a
    cx, cy = (x1a + x2a) // 2, (y1a + y2a) // 2
    side = int(max(bw, bh) * (1.0 + pad_ratio))
    side = max(32, side)
    x1p, y1p = cx - side//2, cy - side//2
    x2p, y2p = x1p + side, y1p + side
    x1p = max(0, min(W-1, x1p)); y1p = max(0, min(H-1, y1p))
    x2p = max(1, min(W, x2p));   y2p = max(1, min(H, y2p))
    return x1p, y1p, x2p, y2p

def draw_panel(frame, lines, color=(0,0,0), bg=(255,255,255)):
    pad = 8; lh = 22
    w = max([cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for t in lines] + [200]) + pad*2
    h = lh*len(lines) + pad*2
    overlay = frame.copy()
    cv2.rectangle(overlay, (10,10), (10+w, 10+h), bg, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    y = 10 + pad + 16
    for t in lines:
        cv2.putText(frame, t, (10+pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += lh

def draw_textbox(frame, header, text, max_width_ratio=0.9, color=(255,255,255), txt=(0,0,0)):
    H, W = frame.shape[:2]
    max_w = int(W * max_width_ratio)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.7, 2
    pad = 10
    line_h = 24

    # simple width-based wrap
    wrapped = []
    cur = ""
    for ch in text if text else "":
        nxt = cur + ch
        w, _ = cv2.getTextSize(nxt, font, fs, th)[0]
        if w > max_w - 2*pad:
            wrapped.append(cur)
            cur = ch
        else:
            cur = nxt
    if cur or not text:
        wrapped.append(cur)

    lines = [header] + wrapped
    box_w = max([cv2.getTextSize(l if l else " ", font, fs, th)[0][0] for l in lines] + [200]) + 2*pad
    box_h = line_h * len(lines) + 2*pad

    x1 = (W - box_w) // 2
    y1 = H - box_h - 10
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = y1 + pad + 16
    for i, l in enumerate(lines):
        if i == 0:
            cv2.putText(frame, l, (x1+pad, y), font, 0.75, (0,0,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, l if l else " ", (x1+pad, y), font, fs, txt, th, cv2.LINE_AA)
        y += line_h

# --------------------------
# Init TTS
# --------------------------
tts = TTSWrapper(rate=args.tts_rate, volume=args.tts_volume, enabled=bool(args.tts))
if args.tts and not tts.enabled:
    print("TTS requested but pyttsx3 not available/failed to init. Install with: pip install pyttsx3")

# --------------------------
# Loop
# --------------------------
cap = cv2.VideoCapture(args.cam)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {args.cam}")

pred_queue = collections.deque(maxlen=max(1, args.smooth))
fps_clock = collections.deque(maxlen=30)
flip_view = True
show_box  = True

# Text builder state
saved_chars = []

print("Controls: q=quit  f=flip  b=box  s=save-char  d=delete  c=clear  e=speak+print")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=args.min_det,
    min_tracking_confidence=args.min_track
) as hands:

    while True:
        t0 = time.time()
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if flip_view:
            frame_bgr = cv2.flip(frame_bgr, 1)

        H, W = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Detect hand
        res = hands.process(frame_rgb)
        crop_rgb = None
        box = None
        hand_present = False

        if res.multi_hand_landmarks:
            hand_present = True
            lm = res.multi_hand_landmarks[0].landmark
            x1, y1, x2, y2 = xywh_from_landmarks(lm, W, H, pad_ratio=args.pad)
            box = (x1, y1, x2, y2)
            crop_rgb = frame_rgb[y1:y2, x1:x2]

        # When no hand is detected, we DO NOT run the model. This prevents spurious predictions.
        if not hand_present:
            # clear smoothing buffer so the old predictions don't persist
            pred_queue.clear()
            label = "(no hand)"
            conf = 0.0
            avg_probs = None
        else:
            # safety: if crop is empty for some reason, use full frame
            if crop_rgb is None or crop_rgb.size == 0:
                crop_rgb = frame_rgb
                box = None

            batch = preprocess_rgb(crop_rgb)
            probs = predict_probs(batch)
            pred_queue.append(probs)
            avg_probs = np.mean(np.stack(pred_queue, 0), 0)
            label, conf = label_from_probs(avg_probs)

        # Draw hand box
        if show_box and box is not None:
            cv2.rectangle(frame_bgr, (box[0],box[1]), (box[2],box[3]), (0,255,255), 2)

        # FPS + HUD
        fps_clock.append(1.0 / max(1e-6, (time.time()-t0)))
        fps = sum(fps_clock)/len(fps_clock)
        hud = [
            f"Pred: {label} ({conf:.2f})",
            f"FPS: {fps:.1f}",
            "Keys: q quit | f flip | b box | s save | d del | c clear | e speak"
        ]
        if not hand_present:
            hud.insert(0, "No hand detected — predictions paused")
        draw_panel(frame_bgr, hud)

        # Saved text box
        text_str = "".join(saved_chars)
        draw_textbox(frame_bgr, header="Saved Text:", text=text_str)

        # Key handling
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('f'):
            flip_view = not flip_view
        elif k == ord('b'):
            show_box = not show_box
        elif k == ord('s'):
            # only allow saving when a hand is present and confidence threshold passed
            if hand_present:
                to_add = SPECIAL_MAP.get(label, label)
                if conf >= args.min_conf and to_add is not None:
                    saved_chars.extend(list(to_add))
            else:
                # optional: give short visual/text feedback; here we print to console
                print("Cannot save: no hand detected.")
        elif k == ord('d'):
            if saved_chars:
                saved_chars.pop()
        elif k == ord('c'):
            saved_chars.clear()
        elif k == ord('e'):
            # speak and print the string
            out = "".join(saved_chars).strip()
            print("Saved string:", out if out else "(empty)")
            if out and args.tts and tts.enabled:
                tts.speak(out)

        cv2.imshow("ASL Live (MediaPipe Hands) + Text Builder + TTS", frame_bgr)

cap.release()
cv2.destroyAllWindows()
try:
    tts.shutdown()
except Exception:
    pass


