import unittest
import os
import io
import cv2
import numpy as np
from app import app
from models.detect_face import detect_face_shape
from models.recommendator import get_recommendation

class TestFaceShapeDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(os.path.dirname(__file__), 'test_images')
        os.makedirs(cls.test_dir, exist_ok=True)
        blank_image_path = os.path.join(cls.test_dir, 'blank.jpg')
        if not os.path.exists(blank_image_path):
            blank = np.zeros((500, 500, 3), dtype=np.uint8)
            cv2.imwrite(blank_image_path, blank)
        cls.blank_image = blank_image_path

    def test_image_not_found(self):
        result = detect_face_shape('nonexistent.jpg')
        self.assertEqual(result, 'Image not found')

    def test_no_face_detected(self):
        result = detect_face_shape(self.blank_image)
        self.assertIn(result, ['No face detected', 'Image not found'])

    def test_detect_various_shapes(self):
        for fname in os.listdir(self.test_dir):
            if fname.lower() == 'blank.jpg':
                continue
            path = os.path.join(self.test_dir, fname)
            expected = os.path.splitext(fname)[0].replace('-', '_').title()
            with self.subTest(image=fname, expected=expected):
                detected = detect_face_shape(path)
                self.assertEqual(detected, expected,
                                 f"Expected {expected}, got {detected} for {fname}")

    def test_face_shape_scores(self):
        # Test specific images that are known to produce certain face shapes
        known_shapes = {
            'oval_image.jpg': 'Oval',
            'square_image.jpg': 'Square',
            'round_image.jpg': 'Round',
            'diamond_image.jpg': 'Diamond',
            'heart_image.jpg': 'Heart',
            'rectangle_image.jpg': 'Rectangle'
        }
        for image, expected_shape in known_shapes.items():
            path = os.path.join(self.test_dir, image)
            if os.path.exists(path):
                detected_shape = detect_face_shape(path)
                self.assertEqual(detected_shape, expected_shape,
                                 f"Expected {expected_shape}, got {detected_shape} for {image}")

class TestRecommendation(unittest.TestCase):
    def test_known_shapes(self):
        for shape in ['Oval', 'Square', 'Round', 'Diamond', 'Heart', 'Rectangle']:
            rec = get_recommendation(shape)
            self.assertIsInstance(rec, dict)
            self.assertIn('hairstyle', rec)
            self.assertIn('beard', rec)

    def test_unknown_shape(self):
        rec = get_recommendation('UnknownShape')
        self.assertEqual(rec['hairstyle'], 'Unknown')
        self.assertEqual(rec['beard'], 'Unknown')

    def test_recommendation_content(self):
        # Test the content of recommendations for known shapes
        recommendations = {
            'Oval': {
                'hairstyle': 'Layered cuts, pompadour, quiff',
                'beard': 'Short boxed beard, stubble'
            },
            'Square': {
                'hairstyle': 'Undercut, side part, buzz cut',
                'beard': 'Circle beard, goatee'
            },
            'Round': {
                'hairstyle': 'Spiky hair, faux hawk',
                'beard': 'Extended goatee, anchor beard'
            },
            'Diamond': {
                'hairstyle': 'Fringe, textured crop',
                'beard': 'Full beard, chinstrap'
            },
            'Heart': {
                'hairstyle': 'Side part, swept-back styles',
                'beard': 'Light stubble, no beard'
            },
            'Rectangle': {
                'hairstyle': 'Classic side part, medium-length styles',
                'beard': 'Full beard to balance face'
            }
        }
        for shape, expected in recommendations.items():
            rec = get_recommendation(shape)
            self.assertEqual(rec, expected)

class TestFlaskApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()
        # Ensure upload folder exists and is empty
        upload_dir = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_dir, exist_ok=True)
        for f in os.listdir(upload_dir):
            os.remove(os.path.join(upload_dir, f))

    def test_index_route(self):
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'<form', res.data)

    def test_upload_no_file(self):
        res = self.client.post('/upload', data={})
        self.assertEqual(res.status_code, 400)
        self.assertIn(b'No file uploaded', res.data)

    def test_upload_invalid_file(self):
        data = {'file': (io.BytesIO(b'data'), 'test.txt')}
        res = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 400)
        self.assertIn(b'Invalid file type', res.data)

    def test_capture_invalid(self):
        res = self.client.post('/capture', data={})
        self.assertEqual(res.status_code, 500)
        self.assertIn(b'Error capturing image', res.data)

    def test_video_feed(self):
        res = self.client.get('/video_feed')
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.content_type.startswith('multipart/x-mixed-replace'))

    def test_result_route(self):
        # Test the result route with a valid image
        valid_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'test_image.jpg')
        # Create a dummy image for testing
        dummy_image = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.imwrite(valid_image_path, dummy_image)

        res = self.client.get('/result?filename=test_image.jpg')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Face Shape:', res.data)

    def test_result_route_no_file(self):
        res = self.client.get('/result?filename=nonexistent.jpg')
        self.assertEqual(res.status_code, 404)
        self.assertIn(b'File not found', res.data)

if __name__ == '__main__':
    unittest.main()
