import os
os.environ["KERAS_BACKEND"] = "torch"

import unittest
import numpy as np
import cv2
import tempfile
import sqlite3
import json

# Import system code
import database as db
import predict as pr
import change_detection as cd
from app import generate_report_pdf

class TestSatelliteSystem(unittest.TestCase):
    
    def setUp(self):
        # Create a dummy image for testing (64x64x3 random RGB image)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dummy_img_path = os.path.join(self.temp_dir.name, "test_sat_image.jpg")
        
        # Draw some mock geographic shapes (green forest block and blue river line)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[10:40, 10:40] = [34, 139, 34] # Forest Green block
        img[45:55, :] = [30, 144, 255]    # River Blue strip
        cv2.imwrite(self.dummy_img_path, img)
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_database_operations(self):
        print("\n--> Testing Database Operations...")
        # 1. Save Prediction
        inserted_id = db.save_prediction(
            image_name="test_sat_image.jpg",
            prediction="Forest",
            confidence=95.4,
            image_url="/static/uploads/dummy.jpg",
            heatmap_url="/static/heatmaps/dummy_heatmap.jpg"
        )
        self.assertIsNotNone(inserted_id)
        self.assertTrue(inserted_id > 0)
        
        # 2. Get History
        history = db.get_history()
        self.assertTrue(len(history) > 0)
        latest_record = history[0]
        self.assertEqual(latest_record["prediction"], "Forest")
        self.assertEqual(latest_record["confidence"], 95.4)
        
        # 3. Get Analytics
        stats = db.get_analytics_stats()
        self.assertTrue(stats["total_predictions"] > 0)
        self.assertEqual(stats["class_distribution"]["Forest"], stats["class_distribution"].get("Forest", 1))
        
        # 4. Delete Record
        success = db.delete_prediction(inserted_id)
        self.assertTrue(success)
        
    def test_change_detection_rules(self):
        print("\n--> Testing Change Detection Rules...")
        # Rule: Forest -> Residential = Deforestation
        res1 = cd.analyze_change("Forest", "Residential")
        self.assertTrue(res1["change_detected"])
        self.assertIn("Deforestation", res1["impact"])
        
        # Rule: Forest -> Industrial = Industrial Expansion
        res2 = cd.analyze_change("Forest", "Industrial")
        self.assertTrue(res2["change_detected"])
        self.assertIn("Industrial Expansion", res2["impact"])
        
        # Rule: Crop -> Industrial = Agricultural Land Loss
        res3 = cd.analyze_change("AnnualCrop", "Industrial")
        self.assertTrue(res3["change_detected"])
        self.assertIn("Agricultural Land Loss", res3["impact"])
        
        # Rule: River -> Industrial = Water Resource Risk
        res4 = cd.analyze_change("River", "Industrial")
        self.assertTrue(res4["change_detected"])
        self.assertIn("Water Resource Risk", res4["impact"])
        
        # Rule: No change = Stable
        res5 = cd.analyze_change("Forest", "Forest")
        self.assertFalse(res5["change_detected"])
        self.assertEqual(res5["impact"], "Stable Land Use")

    def test_prediction_and_gradcam(self):
        print("\n--> Testing Model Inference and Grad-CAM Heatmap Generation...")
        # Initialize model (if not trained, will load with default ImageNet base)
        model = pr.get_or_load_model()
        self.assertIsNotNone(model)
        
        # Run prediction
        res = pr.predict_single_image(self.dummy_img_path)
        self.assertIn("prediction", res)
        self.assertIn("confidence", res)
        self.assertEqual(len(res["top_predictions"]), 5)
        
        # Run Grad-CAM heatmap generation
        heatmap_out_path = os.path.join(self.temp_dir.name, "heatmap_output.jpg")
        pr.generate_gradcam_heatmap(self.dummy_img_path, heatmap_out_path)
        self.assertTrue(os.path.exists(heatmap_out_path))
        self.assertTrue(os.path.getsize(heatmap_out_path) > 0)
        
    def test_report_pdf_generation(self):
        print("\n--> Testing ReportLab PDF Generation...")
        # Mock database record
        mock_record = {
            "id": 999,
            "image_name": "test_sat_image.jpg",
            "prediction": "Forest",
            "confidence": 98.76,
            "timestamp": "2026-06-20 12:00:00",
            "image_url": self.dummy_img_path, # relative url resolving to the file path directly for mock test
            "heatmap_url": self.dummy_img_path
        }
        
        # Create temp file path for pdf
        pdf_out_path = os.path.join(self.temp_dir.name, "test_report.pdf")
        
        # Generate
        generate_report_pdf(mock_record, pdf_out_path)
        self.assertTrue(os.path.exists(pdf_out_path))
        self.assertTrue(os.path.getsize(pdf_out_path) > 0)

if __name__ == "__main__":
    unittest.main()
