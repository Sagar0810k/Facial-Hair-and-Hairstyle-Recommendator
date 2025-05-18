import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# More comprehensive facial landmark indices based on MediaPipe Face Mesh topology
FACE_LANDMARKS = {
    "forehead": 10,      # Top of forehead
    "chin": 152,         # Bottom of chin
    "left_cheek": 123,   # Left cheekbone
    "right_cheek": 352,  # Right cheekbone
    "left_jaw": 234,     # Left jawline
    "right_jaw": 454,    # Right jawline
    "left_temple": 54,   # Left temple
    "right_temple": 284, # Right temple
    "nose_tip": 1,       # Tip of nose
    # Additional points for better accuracy
    "left_cheekbone_high": 111,  # Higher point on left cheekbone
    "right_cheekbone_high": 340, # Higher point on right cheekbone
    "face_mid_left": 67,         # Middle left side of face
    "face_mid_right": 297,       # Middle right side of face
    "jawline_middle": 199        # Middle of jawline
}

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def detect_face_shape(image_path, save_debug_image=True):
    """
    Detect face shape from an image with improved accuracy
    
    Args:
        image_path: Path to the input image
        save_debug_image: Whether to save a debug visualization image
        
    Returns:
        Face shape classification (String)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return "Image not found"
    
    # Convert to RGB (MediaPipe requires RGB input)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    
    # Process with MediaPipe
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
        
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return "No face detected"
        
        # Extract landmarks
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
        
        # Get key facial points
        forehead = landmarks[FACE_LANDMARKS["forehead"]]
        chin = landmarks[FACE_LANDMARKS["chin"]]
        left_cheek = landmarks[FACE_LANDMARKS["left_cheek"]]
        right_cheek = landmarks[FACE_LANDMARKS["right_cheek"]]
        left_jaw = landmarks[FACE_LANDMARKS["left_jaw"]]
        right_jaw = landmarks[FACE_LANDMARKS["right_jaw"]]
        left_temple = landmarks[FACE_LANDMARKS["left_temple"]]
        right_temple = landmarks[FACE_LANDMARKS["right_temple"]]
        nose_tip = landmarks[FACE_LANDMARKS["nose_tip"]]
        left_cheekbone_high = landmarks[FACE_LANDMARKS["left_cheekbone_high"]]
        right_cheekbone_high = landmarks[FACE_LANDMARKS["right_cheekbone_high"]]
        face_mid_left = landmarks[FACE_LANDMARKS["face_mid_left"]]
        face_mid_right = landmarks[FACE_LANDMARKS["face_mid_right"]]
        jawline_middle = landmarks[FACE_LANDMARKS["jawline_middle"]]
        
        # Calculate key distances
        face_length = euclidean_distance(forehead, chin)
        cheekbone_width = euclidean_distance(left_cheek, right_cheek)
        cheekbone_high_width = euclidean_distance(left_cheekbone_high, right_cheekbone_high)
        jaw_width = euclidean_distance(left_jaw, right_jaw)
        temple_width = euclidean_distance(left_temple, right_temple)
        mid_face_width = euclidean_distance(face_mid_left, face_mid_right)
        
        # Calculate vertical distances for diamond shape detection
        forehead_to_cheekbone = euclidean_distance(
            ((left_temple[0] + right_temple[0]) // 2, (left_temple[1] + right_temple[1]) // 2),
            ((left_cheek[0] + right_cheek[0]) // 2, (left_cheek[1] + right_cheek[1]) // 2)
        )
        cheekbone_to_chin = euclidean_distance(
            ((left_cheek[0] + right_cheek[0]) // 2, (left_cheek[1] + right_cheek[1]) // 2),
            chin
        )
        
        # Calculate ratios for classification
        length_to_width = face_length / cheekbone_width
        jaw_to_cheek = jaw_width / cheekbone_width
        temple_to_jaw = temple_width / jaw_width
        temple_to_cheek = temple_width / cheekbone_width
        mid_to_cheek = mid_face_width / cheekbone_width
        cheekbone_to_temple_ratio = cheekbone_width / temple_width
        
        # Additional diamond shape metrics
        forehead_width_ratio = temple_width / cheekbone_width
        cheekbone_prominence = cheekbone_width / jaw_width
        
        # Print measurements for debugging
        print(f"Face Length: {face_length:.2f}")
        print(f"Cheekbone Width: {cheekbone_width:.2f}")
        print(f"Higher Cheekbone Width: {cheekbone_high_width:.2f}")
        print(f"Jaw Width: {jaw_width:.2f}")
        print(f"Temple Width: {temple_width:.2f}")
        print(f"Mid Face Width: {mid_face_width:.2f}")
        print(f"Length-to-Width Ratio: {length_to_width:.2f}")
        print(f"Jaw-to-Cheek Ratio: {jaw_to_cheek:.2f}")
        print(f"Temple-to-Jaw Ratio: {temple_to_jaw:.2f}")
        print(f"Temple-to-Cheek Ratio: {temple_to_cheek:.2f}")
        print(f"Forehead Width Ratio: {forehead_width_ratio:.2f}")
        print(f"Cheekbone Prominence: {cheekbone_prominence:.2f}")
        
        if save_debug_image:
            # Create visualization
            debug_image = rgb_image.copy()
            
            # Draw landmarks with different colors for key points
            mp_drawing.draw_landmarks(
                image=debug_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1)
            )
            
            # Draw key points in different colors
            key_points = {
                "forehead": (forehead, (255, 0, 0)),     # Red
                "chin": (chin, (0, 0, 255)),             # Blue
                "left_cheek": (left_cheek, (0, 255, 0)), # Green
                "right_cheek": (right_cheek, (0, 255, 0)), # Green
                "left_jaw": (left_jaw, (255, 255, 0)),   # Yellow
                "right_jaw": (right_jaw, (255, 255, 0)), # Yellow
                "left_temple": (left_temple, (255, 0, 255)), # Magenta
                "right_temple": (right_temple, (255, 0, 255)), # Magenta
                "left_cheekbone_high": (left_cheekbone_high, (0, 180, 255)), # Light blue
                "right_cheekbone_high": (right_cheekbone_high, (0, 180, 255)), # Light blue
                "face_mid_left": (face_mid_left, (180, 120, 31)), # Brown
                "face_mid_right": (face_mid_right, (180, 120, 31)), # Brown
                "jawline_middle": (jawline_middle, (0, 255, 255)) # Cyan
            }
            
            for point_name, (point, color) in key_points.items():
                cv2.circle(debug_image, point, 5, color, -1)
                cv2.putText(debug_image, point_name, 
                           (point[0] + 5, point[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw measurement lines
            cv2.line(debug_image, forehead, chin, (255, 0, 0), 2)  # Face length
            cv2.line(debug_image, left_cheek, right_cheek, (0, 255, 0), 2)  # Cheekbone width
            cv2.line(debug_image, left_jaw, right_jaw, (255, 255, 0), 2)  # Jaw width
            cv2.line(debug_image, left_temple, right_temple, (255, 0, 255), 2)  # Temple width
            cv2.line(debug_image, face_mid_left, face_mid_right, (180, 120, 31), 2)  # Mid face width
            
            # Draw additional metrics text
            metrics_text = [
                f"Face Shape: TBD",
                f"Length-Width: {length_to_width:.2f}",
                f"Jaw-Cheek: {jaw_to_cheek:.2f}",
                f"Temple-Jaw: {temple_to_jaw:.2f}",
                f"Cheekbone Prominence: {cheekbone_prominence:.2f}"
            ]
            
            for i, text in enumerate(metrics_text):
                cv2.putText(debug_image, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save the debug image
            debug_path = os.path.join(os.path.dirname(image_path), f"debug_{os.path.basename(image_path)}")
            cv2.imwrite(debug_path, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
        
        # Improved classification logic with more nuanced detection for diamond and round faces
        
        # Diamond face detection
        diamond_score = 0
        if cheekbone_prominence > 1.15:  # Prominent cheekbones
            diamond_score += 1
        if jaw_to_cheek < 0.85:  # Narrow jaw compared to cheekbones
            diamond_score += 1
        if forehead_width_ratio < 0.95:  # Narrower forehead compared to cheekbones
            diamond_score += 1
        if temple_to_cheek < 0.98:  # Temples narrower than cheekbones
            diamond_score += 1
            
        # Round face detection
        round_score = 0
        if length_to_width < 1.2:  # Face almost as wide as it is long
            round_score += 1
        if jaw_to_cheek > 0.9:  # Jaw width close to cheekbone width
            round_score += 1
        if temple_to_jaw > 0.95 and temple_to_jaw < 1.05:  # Similar widths throughout
            round_score += 1
        if mid_to_cheek > 0.9:  # Full cheeks
            round_score += 1
            
        # Heart shape detection
        heart_score = 0
        if jaw_to_cheek < 0.8:  # Very narrow jaw compared to cheekbones
            heart_score += 1
        if forehead_width_ratio > 1.0:  # Wider forehead
            heart_score += 1
            
        # Oval face detection
        oval_score = 0
        if length_to_width > 1.35:  # Longer face
            oval_score += 1
        if jaw_to_cheek > 0.8 and jaw_to_cheek < 0.95:  # Moderate jaw
            oval_score += 1
        if cheekbone_prominence < 1.15 and cheekbone_prominence > 0.95:  # Balanced cheekbones
            oval_score += 1
            
        # Square face detection
        square_score = 0
        if length_to_width < 1.2:  # Face almost as wide as it is long
            square_score += 1
        if jaw_to_cheek > 0.95:  # Jaw width similar to cheekbone width
            square_score += 1
        if temple_to_jaw > 0.95 and temple_to_jaw < 1.05:  # Similar width at temples and jaw
            square_score += 1
            
        # Rectangle face detection
        rectangle_score = 0
        if length_to_width > 1.3:  # Longer face
            rectangle_score += 1
        if jaw_to_cheek > 0.9:  # Wider jaw
            rectangle_score += 1
        if temple_to_jaw > 0.9 and temple_to_jaw < 1.1:  # Similar width at temples and jaw
            rectangle_score += 1
            
        # Get the highest scoring face shape
        scores = {
            "Diamond": diamond_score,
            "Round": round_score,
            "Heart": heart_score,
            "Oval": oval_score,
            "Square": square_score,
            "Rectangle": rectangle_score
        }
        
        # Print all scores for debugging
        print("Face shape scores:", scores)
        
        # Get the face shape with the highest score
        face_shape = max(scores, key=scores.get)
        
        # Handle ties with preference for diamond and round over rectangle if requested
        if scores[face_shape] == scores["Rectangle"] and face_shape != "Rectangle":
            # Keep the non-Rectangle classification
            pass
        elif scores["Diamond"] == scores["Rectangle"] or scores["Round"] == scores["Rectangle"]:
            # If you mentioned diamond or round in your request, prioritize those
            if scores["Diamond"] == scores["Rectangle"]:
                face_shape = "Diamond"
            else:
                face_shape = "Round"
                
        print(f"Detected face shape: {face_shape}")
        return face_shape


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "face_image.jpg"
    face_shape = detect_face_shape(image_path, save_debug_image=True)
    print(f"Face shape: {face_shape}")