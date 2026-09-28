import os
import logging

logging.basicConfig(level=logging.INFO)
os.environ["KERAS_BACKEND"] = "torch"

from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS
import uuid
import json
from datetime import datetime

# Import project modules
import database as db
import train as tr
import predict as pr
import change_detection as cd

app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app)

# Folders setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
HEATMAP_FOLDER = os.path.join(BASE_DIR, "static", "heatmaps")
REPORTS_FOLDER = os.path.join(BASE_DIR, "static", "reports")

for folder in [UPLOAD_FOLDER, HEATMAP_FOLDER, REPORTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Helper to save files with unique names
def save_uploaded_file(file, folder):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        raise ValueError("Unsupported file type. Use JPG or PNG.")
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(folder, filename)
    file.save(filepath)
    return filename, filepath

@app.route("/train", methods=["POST"])
def api_train():
    try:
        # Determine if quick training is requested. Default to True for demonstration safety
        data = request.get_json(silent=True) or {}
        quick = data.get("quick", True)
        
        metrics = tr.train_model(quick_demo=quick)
        # Invalidate cached model so next predictions use the newly trained model
        try:
            import predict as pr
            pr._model_cache = None
        except Exception:
            pass
        
        return jsonify({
            "status": "success",
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
@app.route("/predict", methods=["POST"])
def api_predict():
    try:
        logging.info("Received /predict request")
        uploaded_file = request.files.get("image") or request.files.get("file")
        if uploaded_file is None:
            logging.error("No image file provided in request")
            return jsonify({"status": "error", "message": "No image file provided"}), 400
        
        if uploaded_file.filename == "":
            logging.error("Empty filename in uploaded file")
            return jsonify({"status": "error", "message": "Empty file name"}), 400
        
        # Save uploaded image
        filename, filepath = save_uploaded_file(uploaded_file, UPLOAD_FOLDER)
        logging.info(f"Saved uploaded file as {filename}")
        
        # Run classification
        results = pr.predict_single_image(filepath)
        logging.info(f"Prediction results: {results}")
        
        # Generate Grad-CAM heatmap
        heatmap_filename = f"gradcam_{filename}"
        heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)
        pr.generate_gradcam_heatmap(filepath, heatmap_path)
        logging.info(f"Generated Grad-CAM heatmap at {heatmap_path}")
        
        # Relative URLs
        image_url = f"/static/uploads/{filename}"
        heatmap_url = f"/static/heatmaps/{heatmap_filename}"
        
        # Log to Database
        inserted_id = db.save_prediction(
            image_name=uploaded_file.filename,
            prediction=results["prediction"],
            confidence=results["confidence"],
            image_url=image_url,
            heatmap_url=heatmap_url
        )
        logging.info(f"Saved prediction to DB with id {inserted_id}")
        
        return jsonify({
            "id": inserted_id,
            "prediction": results["prediction"],
            "confidence": results["confidence"],
            "top_predictions": results["top_predictions"],
            "image_url": image_url,
            "heatmap_url": heatmap_url
        })
    except Exception as e:
        logging.exception("Error during /predict")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/batch_predict", methods=["POST"])
def api_batch_predict():
    try:
        logging.info(f"Received batch_predict request. Keys in request.files: {list(request.files.keys())}")
        if "images" not in request.files:
            # Check for multiple files field
            files = request.files.getlist("images[]")
            logging.info(f"Found images[]: retrieved {len(files)} files")
            if not files:
                return jsonify({"status": "error", "message": "No image files provided"}), 400
        else:
            files = request.files.getlist("images")
            logging.info(f"Found images: retrieved {len(files)} files")
            
        predictions = []
        for file in files:
            if file.filename == "":
                continue
                
            try:
                # Save and Predict
                filename, filepath = save_uploaded_file(file, UPLOAD_FOLDER)
                results = pr.predict_single_image(filepath)
                
                # Generate Grad-CAM heatmap with fallback
                heatmap_filename = f"gradcam_{filename}"
                heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)
                try:
                    pr.generate_gradcam_heatmap(filepath, heatmap_path)
                except Exception as hm_err:
                    logging.warning(f"Grad-CAM generation failed: {hm_err}, generating dummy heatmap")
                    from predict import generate_dummy_heatmap
                    generate_dummy_heatmap(cv2.imread(filepath), heatmap_path)
                
                image_url = f"/static/uploads/{filename}"
                heatmap_url = f"/static/heatmaps/{heatmap_filename}"
                
                # Log to DB
                inserted_id = db.save_prediction(
                    image_name=file.filename,
                    prediction=results["prediction"],
                    confidence=results["confidence"],
                    image_url=image_url,
                    heatmap_url=heatmap_url
                )
                
                predictions.append({
                    "id": inserted_id,
                    "image_name": file.filename,
                    "prediction": results["prediction"],
                    "confidence": results["confidence"],
                    "image_url": image_url,
                    "heatmap_url": heatmap_url
                })
            except Exception as item_error:
                predictions.append({
                    "image_name": file.filename,
                    "status": "error",
                    "message": str(item_error)
                })
                
        return jsonify({
            "status": "success",
            "predictions": predictions
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/change_detection", methods=["POST"])
def api_change_detection():
    try:
        if "old_image" not in request.files or "new_image" not in request.files:
            return jsonify({"status": "error", "message": "Both old_image and new_image are required"}), 400
            
        old_file = request.files["old_image"]
        new_file = request.files["new_image"]
        
        if old_file.filename == "" or new_file.filename == "":
            return jsonify({"status": "error", "message": "Empty file names uploaded"}), 400
            
        # Save images
        old_filename, old_filepath = save_uploaded_file(old_file, UPLOAD_FOLDER)
        new_filename, new_filepath = save_uploaded_file(new_file, UPLOAD_FOLDER)
        
        # Predict both
        old_res = pr.predict_single_image(old_filepath)
        new_res = pr.predict_single_image(new_filepath)
        
        # Analyze Change
        analysis = cd.analyze_change(old_res["prediction"], new_res["prediction"])
        
        # Build URLs
        old_image_url = f"/static/uploads/{old_filename}"
        new_image_url = f"/static/uploads/{new_filename}"
        
        # Generate heatmaps
        old_hm_filename = f"gradcam_{old_filename}"
        new_hm_filename = f"gradcam_{new_filename}"
        pr.generate_gradcam_heatmap(old_filepath, os.path.join(HEATMAP_FOLDER, old_hm_filename))
        pr.generate_gradcam_heatmap(new_filepath, os.path.join(HEATMAP_FOLDER, new_hm_filename))
        
        # Save predictions to history too
        db.save_prediction(old_file.filename, old_res["prediction"], old_res["confidence"], old_image_url, f"/static/heatmaps/{old_hm_filename}")
        db.save_prediction(new_file.filename, new_res["prediction"], new_res["confidence"], new_image_url, f"/static/heatmaps/{new_hm_filename}")
        
        return jsonify({
            "old_class": analysis["old_class"],
            "new_class": analysis["new_class"],
            "change_detected": analysis["change_detected"],
            "impact": analysis["impact"],
            "old_confidence": old_res["confidence"],
            "new_confidence": new_res["confidence"],
            "old_image_url": old_image_url,
            "new_image_url": new_image_url,
            "old_heatmap_url": f"/static/heatmaps/{old_hm_filename}",
            "new_heatmap_url": f"/static/heatmaps/{new_hm_filename}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/history", methods=["GET"])
def api_history():
    try:
        history = db.get_history()
        return jsonify(history)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/history/<int:pred_id>", methods=["DELETE"])
def api_delete_history(pred_id):
    try:
        db.delete_prediction(pred_id)
        return jsonify({"status": "success", "message": "Record deleted"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/analytics", methods=["GET"])
def api_analytics():
    try:
        stats = db.get_analytics_stats()
        # Add model training stats (from metrics.json if available)
        model_metrics = tr.get_default_metrics()
        stats["model_metrics"] = {
            "accuracy": model_metrics["accuracy"],
            "precision": model_metrics["precision"],
            "recall": model_metrics["recall"],
            "f1_score": model_metrics["f1_score"],
            "confusion_matrix": model_metrics["confusion_matrix"],
            "history": model_metrics["history"]
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/download_report/<int:pred_id>", methods=["GET"])
def api_download_report(pred_id):
    try:
        conn = db.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return jsonify({"status": "error", "message": "Prediction record not found"}), 404
            
        record = dict(row)
        
        # Create ReportLab PDF
        report_filename = f"report_{pred_id}_{uuid.uuid4().hex[:8]}.pdf"
        report_path = os.path.join(REPORTS_FOLDER, report_filename)
        
        generate_report_pdf(record, report_path)
        
        return send_file(report_path, as_attachment=True, download_name=f"satellite_report_{pred_id}.pdf")
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def generate_report_pdf(record, pdf_path):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Custom colors
    primary_color = colors.HexColor("#0B3C5D") # Sleek GIS Blue
    secondary_color = colors.HexColor("#328CC1")
    text_color = colors.HexColor("#1D2731")
    light_bg = colors.HexColor("#F9F9F9")
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=primary_color,
        spaceAfter=15,
        alignment=1 # Centered
    )
    
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=secondary_color,
        spaceBefore=12,
        spaceAfter=6,
        borderPadding=2
    )
    
    label_style = ParagraphStyle(
        'LabelStyle',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica-Bold',
        textColor=primary_color
    )
    
    value_style = ParagraphStyle(
        'ValueStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=text_color
    )
    
    desc_style = ParagraphStyle(
        'DescStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=text_color,
        leading=14
    )

    # 1. Header Bar
    story.append(Paragraph("Satellite Land Classification Report", title_style))
    story.append(Spacer(1, 10))
    
    # 2. Metadata Table
    meta_data = [
        [Paragraph("Record ID:", label_style), Paragraph(str(record['id']), value_style),
         Paragraph("Timestamp:", label_style), Paragraph(record['timestamp'], value_style)],
        [Paragraph("Image Name:", label_style), Paragraph(record['image_name'], value_style),
         Paragraph("Confidence:", label_style), Paragraph(f"{record['confidence']:.2f}%", value_style)],
        [Paragraph("Classification:", label_style), Paragraph(record['prediction'], value_style),
         Paragraph("Model:", label_style), Paragraph("ResNet50V2 (Transfer Learning)", value_style)]
    ]
    t = Table(meta_data, colWidths=[100, 160, 100, 160])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E5E5E5")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#F0F0F0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # 3. Image Layout (Side by Side)
    # Check if images exist on disk
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    orig_path = os.path.join(backend_dir, record['image_url'].lstrip('/'))
    hm_path = os.path.join(backend_dir, record['heatmap_url'].lstrip('/'))
    
    image_row = []
    if os.path.exists(orig_path):
        image_row.append(Image(orig_path, width=220, height=220))
    else:
        image_row.append(Paragraph("[Original Image Missing]", label_style))
        
    if os.path.exists(hm_path):
        image_row.append(Image(hm_path, width=220, height=220))
    else:
        image_row.append(Paragraph("[Grad-CAM Heatmap Missing]", label_style))
        
    image_table = Table([image_row], colWidths=[260, 260])
    image_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(image_table)
    story.append(Spacer(1, 10))
    
    # Image Labels
    label_row = [
        Paragraph("<para align=center><b>Original Satellite Image</b></para>", value_style),
        Paragraph("<para align=center><b>Grad-CAM Explainability Heatmap</b></para>", value_style)
    ]
    label_table = Table([label_row], colWidths=[260, 260])
    story.append(label_table)
    story.append(Spacer(1, 20))
    
    # 4. Explainability and Analysis
    story.append(Paragraph("Land Cover Description", section_style))
    descriptions = {
        "AnnualCrop": "Area designated for crops that are planted and harvested annually (e.g., wheat, maize). These regions are vital for food security but require careful agricultural management to prevent soil degradation.",
        "Forest": "Dense canopy vegetation consisting of trees. Forests act as crucial carbon sinks, preserve biodiversity, and mitigate local temperature rises.",
        "HerbaceousVegetation": "Natural grassland and shrublands. These serve as key grazing fields, support local wildlife habitats, and stabilize soils.",
        "Highway": "Paved roads and transit corridors. High connectivity signals infrastructural development but can lead to habitat fragmentation and local noise pollution.",
        "Industrial": "Manufacturing units, warehouses, and industrial parks. These zones represent high economic activity but carry risks of industrial waste, water contamination, and air quality degradation.",
        "Pasture": "Grassland areas designated for livestock grazing. Soil compaction and runoff of nitrogenous compounds are concerns if overgrazed.",
        "PermanentCrop": "Land used for agricultural growth of trees or shrubs that are not replanted annually (e.g., vineyards, orchards). Provides long-term vegetation cover compared to annual crops.",
        "Residential": "Housing areas, urban settlements, and suburban pockets. Reflects population concentration, urban expansion, and demands local waste management systems.",
        "River": "Lotic freshwater bodies. Crucial supply channels for agriculture, drinking water, and ecology. Vulnerable to erosion and industrial runoff.",
        "SeaLake": "Lentic water bodies and marine areas. Critical ecosystems supporting aquatic biodiversity and modulating regional weather patterns."
    }
    
    pred_desc = descriptions.get(record['prediction'], "Classified land cover type under the EuroSAT dataset categories.")
    story.append(Paragraph(pred_desc, desc_style))
    story.append(Spacer(1, 12))
    
    # 5. Grad-CAM Interpretation
    story.append(Paragraph("Grad-CAM Explainability Analysis", section_style))
    cam_desc = (
        "The Grad-CAM heatmap highlights the specific pixel regions that the deep learning model (ResNet50V2) "
        "focused on to determine the class. Red and yellow regions indicate high-intensity features (like edges, buildings, "
        "or water textures) that heavily influenced the classification, while blue regions represents elements that "
        "had negligible impact. This validation ensures that the model is identifying genuine geographical features "
        "rather than relying on peripheral noise."
    )
    story.append(Paragraph(cam_desc, desc_style))
    story.append(Spacer(1, 20))
    
    # Footer disclaimer
    story.append(Paragraph("<font size=8 color='#888888'>This report is automatically generated by the Satellite Land Classification and Change Detection System dashboard. Classification accuracy is based on training validations on the EuroSAT dataset.</font>", value_style))
    
    doc.build(story)

@app.route("/reset_model", methods=["POST"])
def api_reset_model():
    try:
        # Delete trained model file if exists (safely ignore locks on Windows)
        if os.path.exists(tr.MODEL_PATH):
            try:
                os.remove(tr.MODEL_PATH)
            except Exception as remove_err:
                logging.warning(f"Could not remove model file: {remove_err}. It will be overwritten during training.")
        # Clear model cache in predict module
        try:
            import predict as pr
            pr._model_cache = None
        except Exception:
            pass
        # Optionally retrain quickly to have a model
        metrics = tr.train_model(quick_demo=True)
        return jsonify({"status": "success", "message": "Model reset and retrained (quick demo).", "metrics": metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/grid_predict", methods=["POST"])
def api_grid_predict():
    try:
        uploaded_file = request.files.get("image") or request.files.get("file")
        if uploaded_file is None:
            return jsonify({"status": "error", "message": "No image file provided"}), 400
        
        if uploaded_file.filename == "":
            return jsonify({"status": "error", "message": "Empty file name"}), 400
            
        filename, filepath = save_uploaded_file(uploaded_file, UPLOAD_FOLDER)
        
        results = pr.predict_image_grid(filepath)
        results["image_url"] = f"/static/uploads/{filename}"
        
        return jsonify(results)
    except Exception as e:
        logging.exception("Error in grid predict")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
