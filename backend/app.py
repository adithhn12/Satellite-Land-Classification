import os
import logging

logging.basicConfig(level=logging.INFO)
os.environ["KERAS_BACKEND"] = "torch"

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import uuid

# Import project modules
import database as db
import train as tr
import predict as pr
import change_detection as cd
import eval_external as eval_ext

app = Flask(__name__)
CORS(app)

# Folders setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
HEATMAP_FOLDER = os.path.join(BASE_DIR, "static", "heatmaps")
REPORTS_FOLDER = os.path.join(BASE_DIR, "static", "reports")

for folder in [UPLOAD_FOLDER, HEATMAP_FOLDER, REPORTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def save_uploaded_file(file, folder):
    if file is None or file.filename == "":
        raise ValueError("No file uploaded or filename is empty.")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        raise ValueError("Unsupported file format. Please upload JPG, PNG, or TIFF satellite images.")
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(folder, filename)
    file.save(filepath)
    return filename, filepath

@app.route("/train", methods=["POST"])
def api_train():
    try:
        data = request.get_json(silent=True) or {}
        mode = data.get("mode", "demo") # "production" vs "demo"
        quick = data.get("quick", True)
        
        is_prod = (mode == "production") or (not quick)
        
        metrics = tr.train_model(quick_demo=not is_prod, is_production=is_prod)
        
        # Invalidate cached model
        try:
            pr._model_cache = None
        except Exception:
            pass
            
        return jsonify({
            "status": "success",
            "is_production": is_prod,
            "metrics": metrics
        })
    except Exception as e:
        logging.exception("Error during model training")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict", methods=["POST"])
def api_predict():
    try:
        uploaded_file = request.files.get("image") or request.files.get("file")
        if uploaded_file is None:
            return jsonify({"status": "error", "message": "No image file provided in request."}), 400
            
        filename, filepath = save_uploaded_file(uploaded_file, UPLOAD_FOLDER)
        
        # Classification prediction
        results = pr.predict_single_image(filepath)
        
        # Grad-CAM heatmap generation with graceful handling
        image_url = f"/static/uploads/{filename}"
        heatmap_url = None
        heatmap_filename = f"gradcam_{filename}"
        heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)
        
        try:
            pr.generate_gradcam_heatmap(filepath, heatmap_path)
            heatmap_url = f"/static/heatmaps/{heatmap_filename}"
        except Exception as hm_err:
            logging.warning(f"Grad-CAM generation failed: {hm_err}")
            
        # Log to Database
        inserted_id = db.save_prediction(
            image_name=uploaded_file.filename,
            prediction=results["prediction"],
            confidence=results["confidence"],
            image_url=image_url,
            heatmap_url=heatmap_url
        )
        
        return jsonify({
            "id": inserted_id,
            "prediction": results["prediction"],
            "confidence": results["confidence"],
            "is_low_confidence": results.get("is_low_confidence", False),
            "confidence_label": results.get("confidence_label", "High Confidence"),
            "top_predictions": results["top_predictions"],
            "image_url": image_url,
            "heatmap_url": heatmap_url
        })
    except FileNotFoundError as fnf:
        return jsonify({"status": "error", "message": str(fnf)}), 404
    except Exception as e:
        logging.exception("Error during /predict")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/batch_predict", methods=["POST"])
def api_batch_predict():
    try:
        files = request.files.getlist("images") or request.files.getlist("images[]")
        if not files:
            return jsonify({"status": "error", "message": "No image files provided for batch processing."}), 400
            
        predictions = []
        for file in files:
            if file.filename == "":
                continue
            try:
                filename, filepath = save_uploaded_file(file, UPLOAD_FOLDER)
                results = pr.predict_single_image(filepath)
                
                heatmap_filename = f"gradcam_{filename}"
                heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)
                heatmap_url = None
                try:
                    pr.generate_gradcam_heatmap(filepath, heatmap_path)
                    heatmap_url = f"/static/heatmaps/{heatmap_filename}"
                except Exception:
                    pass
                    
                image_url = f"/static/uploads/{filename}"
                
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
                    "is_low_confidence": results.get("is_low_confidence", False),
                    "image_url": image_url,
                    "heatmap_url": heatmap_url
                })
            except Exception as item_err:
                predictions.append({
                    "image_name": file.filename,
                    "status": "error",
                    "message": str(item_err)
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
            return jsonify({"status": "error", "message": "Both old_image and new_image files are required."}), 400
            
        old_file = request.files["old_image"]
        new_file = request.files["new_image"]
        
        old_filename, old_filepath = save_uploaded_file(old_file, UPLOAD_FOLDER)
        new_filename, new_filepath = save_uploaded_file(new_file, UPLOAD_FOLDER)
        
        old_res = pr.predict_single_image(old_filepath)
        new_res = pr.predict_single_image(new_filepath)
        
        analysis = cd.analyze_change(old_res["prediction"], new_res["prediction"])
        
        old_image_url = f"/static/uploads/{old_filename}"
        new_image_url = f"/static/uploads/{new_filename}"
        
        old_hm_url, new_hm_url = None, None
        try:
            pr.generate_gradcam_heatmap(old_filepath, os.path.join(HEATMAP_FOLDER, f"gradcam_{old_filename}"))
            old_hm_url = f"/static/heatmaps/gradcam_{old_filename}"
        except Exception: pass
        
        try:
            pr.generate_gradcam_heatmap(new_filepath, os.path.join(HEATMAP_FOLDER, f"gradcam_{new_filename}"))
            new_hm_url = f"/static/heatmaps/gradcam_{new_filename}"
        except Exception: pass
        
        db.save_prediction(old_file.filename, old_res["prediction"], old_res["confidence"], old_image_url, old_hm_url)
        db.save_prediction(new_file.filename, new_res["prediction"], new_res["confidence"], new_image_url, new_hm_url)
        
        return jsonify({
            "old_class": analysis["old_class"],
            "new_class": analysis["new_class"],
            "change_detected": analysis["change_detected"],
            "impact": analysis["impact"],
            "old_confidence": old_res["confidence"],
            "new_confidence": new_res["confidence"],
            "old_image_url": old_image_url,
            "new_image_url": new_image_url,
            "old_heatmap_url": old_hm_url,
            "new_heatmap_url": new_hm_url
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/grid_predict", methods=["POST"])
def api_grid_predict():
    try:
        uploaded_file = request.files.get("image") or request.files.get("file")
        if uploaded_file is None:
            return jsonify({"status": "error", "message": "No image file provided."}), 400
            
        filename, filepath = save_uploaded_file(uploaded_file, UPLOAD_FOLDER)
        
        results = pr.predict_image_grid(filepath)
        results["image_url"] = f"/static/uploads/{filename}"
        
        return jsonify(results)
    except FileNotFoundError as fnf:
        return jsonify({"status": "error", "message": str(fnf)}), 404
    except Exception as e:
        logging.exception("Error in grid predict")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/history", methods=["GET"])
def api_history():
    try:
        return jsonify(db.get_history())
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
        # Fetch real production metrics (no hardcoded fake accuracy)
        prod_metrics = tr.get_production_metrics()
        ext_metrics = eval_ext.get_external_metrics()
        
        stats["model_metrics"] = prod_metrics
        stats["external_metrics"] = ext_metrics
        return jsonify(stats)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/eval_external", methods=["GET", "POST"])
def api_eval_external():
    try:
        results = eval_ext.evaluate_external_dataset()
        return jsonify(results)
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
    
    primary_color = colors.HexColor("#0B3C5D")
    secondary_color = colors.HexColor("#328CC1")
    text_color = colors.HexColor("#1D2731")
    light_bg = colors.HexColor("#F9F9F9")
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=22, textColor=primary_color, spaceAfter=15, alignment=1)
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontSize=13, textColor=secondary_color, spaceBefore=12, spaceAfter=6)
    label_style = ParagraphStyle('LabelStyle', parent=styles['Normal'], fontSize=9, fontName='Helvetica-Bold', textColor=primary_color)
    value_style = ParagraphStyle('ValueStyle', parent=styles['Normal'], fontSize=9, textColor=text_color)
    desc_style = ParagraphStyle('DescStyle', parent=styles['Normal'], fontSize=9, textColor=text_color, leading=13)

    story.append(Paragraph("Satellite Land Classification Report", title_style))
    story.append(Spacer(1, 10))
    
    meta_data = [
        [Paragraph("Record ID:", label_style), Paragraph(str(record['id']), value_style), Paragraph("Timestamp:", label_style), Paragraph(str(record['timestamp']), value_style)],
        [Paragraph("Image Name:", label_style), Paragraph(str(record['image_name']), value_style), Paragraph("Confidence:", label_style), Paragraph(f"{record['confidence']:.2f}%", value_style)],
        [Paragraph("Classification:", label_style), Paragraph(str(record['prediction']), value_style), Paragraph("Model Architecture:", label_style), Paragraph("ResNet50V2 Transfer Learning", value_style)]
    ]
    t = Table(meta_data, colWidths=[90, 170, 90, 170])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E5E5E5")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#F0F0F0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))
    
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    orig_path = os.path.join(backend_dir, record['image_url'].lstrip('/')) if record.get('image_url') else None
    hm_path = os.path.join(backend_dir, record['heatmap_url'].lstrip('/')) if record.get('heatmap_url') else None
    
    image_row = []
    if orig_path and os.path.exists(orig_path):
        image_row.append(Image(orig_path, width=200, height=200))
    else:
        image_row.append(Paragraph("[Original Image Unavailable]", label_style))
        
    if hm_path and os.path.exists(hm_path):
        image_row.append(Image(hm_path, width=200, height=200))
    else:
        image_row.append(Paragraph("[Grad-CAM Heatmap Unavailable]", label_style))
        
    image_table = Table([image_row], colWidths=[250, 250])
    image_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(image_table)
    story.append(Spacer(1, 8))
    
    label_row = [
        Paragraph("<para align=center><b>Original Satellite Imagery</b></para>", value_style),
        Paragraph("<para align=center><b>Grad-CAM Explainability Heatmap</b></para>", value_style)
    ]
    story.append(Table([label_row], colWidths=[250, 250]))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("Land Cover Description", section_style))
    descriptions = {
        "AnnualCrop": "Annual crops planted and harvested within a seasonal cycle. Vital for agricultural productivity and food security.",
        "Forest": "Dense trees and vegetation canopy serving as crucial global carbon sinks and biodiversity habitats.",
        "HerbaceousVegetation": "Natural grasslands and wild vegetation used for grazing and habitat protection.",
        "Highway": "Paved highways and transportation infrastructure corridors.",
        "Industrial": "Manufacturing facilities, commercial complexes, and warehousing units.",
        "Pasture": "Managed agricultural grasslands for livestock grazing.",
        "PermanentCrop": "Perennial crops, orchards, and vineyards.",
        "Residential": "Housing, residential communities, and urban developments.",
        "River": "Inland freshwater channels and rivers.",
        "SeaLake": "Marine areas, coastal waters, and inland lakes."
    }
    story.append(Paragraph(descriptions.get(record['prediction'], "Land cover category classified under EuroSAT scheme."), desc_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Grad-CAM Technical Interpretation", section_style))
    story.append(Paragraph("Grad-CAM highlights input pixel regions that contributed most significantly to the neural network's final class prediction probability.", desc_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("<font size=8 color='#888888'>Automated report generated by Satellite GIS Classification System.</font>", value_style))
    doc.build(story)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
