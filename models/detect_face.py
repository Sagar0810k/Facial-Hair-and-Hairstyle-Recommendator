import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils

# Landmark indices…
FACE_LANDMARKS = {
    "forehead": 10, "chin": 152,
    "left_cheek": 123, "right_cheek": 352,
    "left_jaw": 234, "right_jaw": 454,
    "left_temple": 54, "right_temple": 284,
    "nose_tip": 1,
    "left_cheekbone_high": 111, "right_cheekbone_high": 340,
    "face_mid_left": 67, "face_mid_right": 297,
    "jawline_middle": 199
}

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def detect_face_shape(image_path, save_debug_image=True):
    image = cv2.imread(image_path)
    if image is None:
        return "Image not found"
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # 1) Run face‐mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               min_detection_confidence=0.5) as fm:
        res = fm.process(rgb_image)
        if not res.multi_face_landmarks:
            return "No face detected"
        lm = res.multi_face_landmarks[0].landmark
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]

    # 2) Extract key points & distances/ratios (unchanged)
    forehead  = pts[FACE_LANDMARKS["forehead"]]
    chin      = pts[FACE_LANDMARKS["chin"]]
    left_cheek, right_cheek = pts[FACE_LANDMARKS["left_cheek"]], pts[FACE_LANDMARKS["right_cheek"]]
    left_jaw,   right_jaw   = pts[FACE_LANDMARKS["left_jaw"]],   pts[FACE_LANDMARKS["right_jaw"]]
    left_temp,  right_temp  = pts[FACE_LANDMARKS["left_temple"]], pts[FACE_LANDMARKS["right_temple"]]
    left_cb_h,  right_cb_h  = pts[FACE_LANDMARKS["left_cheekbone_high"]], pts[FACE_LANDMARKS["right_cheekbone_high"]]
    mid_l,      mid_r       = pts[FACE_LANDMARKS["face_mid_left"]], pts[FACE_LANDMARKS["face_mid_right"]]
    
    face_length         = euclidean_distance(forehead, chin)
    cheekbone_width     = euclidean_distance(left_cheek, right_cheek)
    jaw_width           = euclidean_distance(left_jaw, right_jaw)
    temple_width        = euclidean_distance(left_temp, right_temp)
    mid_face_width      = euclidean_distance(mid_l, mid_r)
    cheekbone_prom      = cheekbone_width / jaw_width
    length_to_width     = face_length / cheekbone_width
    jaw_to_cheek        = jaw_width / cheekbone_width
    temple_to_cheek     = temple_width / cheekbone_width
    temple_to_jaw       = temple_width / jaw_width
    forehead_width_ratio= temple_width / cheekbone_width

    # 3) Compute all shape‐scores (unchanged logic)
    diamond_score = int(cheekbone_prom > 1.15) + int(jaw_to_cheek < 0.85) \
                  + int(forehead_width_ratio < 0.95) + int(temple_to_cheek < 0.98)
    round_score   = int(length_to_width < 1.2) + int(jaw_to_cheek > 0.9) \
                  + int(0.95 < temple_to_jaw < 1.05) + int(mid_face_width/cheekbone_width > 0.9)
    heart_score   = int(jaw_to_cheek < 0.8) + int(forehead_width_ratio > 1.0)
    oval_score    = int(length_to_width > 1.35) + int(0.8 < jaw_to_cheek < 0.95) \
                  + int(0.95 < cheekbone_prom < 1.15)
    square_score  = int(length_to_width < 1.2) + int(jaw_to_cheek > 0.95) \
                  + int(0.95 < temple_to_jaw < 1.05)
    rect_score    = int(length_to_width > 1.3) + int(jaw_to_cheek > 0.9) \
                  + int(0.9 < temple_to_jaw < 1.1)

    scores = {
        "Diamond": diamond_score,
        "Round":   round_score,
        "Heart":   heart_score,
        "Oval":    oval_score,
        "Square":  square_score,
        "Rectangle": rect_score
    }
    # Choose max, with tie‐breaking favoring Diamond/Round over Rectangle
    face_shape = max(scores, key=lambda k: (scores[k], k in ["Diamond","Round"]))

    # 4) NOW draw debug image **after** we have face_shape
    if save_debug_image:
        dbg = rgb_image.copy()
        mp_drawing.draw_landmarks(
            dbg, res.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
            None, mp_drawing.DrawingSpec(color=(80,110,10), thickness=1))
        # draw your circles/lines here exactly as before…

        # rebuild the metrics text with the real face_shape
        metrics = [
            f"Face Shape: {face_shape}",                 # <-- updated
            f"Length-Width: {length_to_width:.2f}",
            f"Jaw-Cheek: {jaw_to_cheek:.2f}",
            f"Temple-Jaw: {temple_to_jaw:.2f}",
            f"Cheekbone Prom: {cheekbone_prom:.2f}"
        ]
        for i, txt in enumerate(metrics):
            cv2.putText(dbg, txt, (10, 30 + 25*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        out_path = os.path.join(os.path.dirname(image_path),
                                f"debug_{os.path.basename(image_path)}")
        cv2.imwrite(out_path, cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))

    return face_shape

# Example
if __name__ == "__main__":
    print(detect_face_shape("face_image.jpg", save_debug_image=True))
