import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "satellite_system.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_url TEXT,
            heatmap_url TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(image_name, prediction, confidence, image_url=None, heatmap_url=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (image_name, prediction, confidence, image_url, heatmap_url)
        VALUES (?, ?, ?, ?, ?)
    """, (image_name, prediction, confidence, image_url, heatmap_url))
    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id

def get_history():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    history = [dict(row) for row in rows]
    conn.close()
    return history

def delete_prediction(pred_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Optional: fetch paths to clean up files physically
    cursor.execute("SELECT image_url, heatmap_url FROM predictions WHERE id = ?", (pred_id,))
    row = cursor.fetchone()
    if row:
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        # Clean up files if they exist in static folder
        for file_url in [row['image_url'], row['heatmap_url']]:
            if file_url:
                # Remove leading slash or prefix
                clean_path = file_url.lstrip('/')
                abs_path = os.path.join(backend_dir, clean_path)
                if os.path.exists(abs_path) and os.path.isfile(abs_path):
                    try:
                        os.remove(abs_path)
                    except Exception as e:
                        print(f"Error removing file {abs_path}: {e}")
                        
    cursor.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    conn.commit()
    conn.close()
    return True

def get_analytics_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Total Predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]
    
    # 2. Avg Confidence
    cursor.execute("SELECT AVG(confidence) FROM predictions")
    avg_confidence = cursor.fetchone()[0] or 0.0
    
    # 3. Class Distribution
    cursor.execute("SELECT prediction, COUNT(*) as count FROM predictions GROUP BY prediction")
    distribution_rows = cursor.fetchall()
    class_distribution = {row['prediction']: row['count'] for row in distribution_rows}
    
    # 4. History trend (aggregated by date)
    cursor.execute("""
        SELECT DATE(timestamp) as date, COUNT(*) as count 
        FROM predictions 
        GROUP BY DATE(timestamp) 
        ORDER BY date ASC 
        LIMIT 30
    """)
    trend_rows = cursor.fetchall()
    trends = {row['date']: row['count'] for row in trend_rows}
    
    conn.close()
    
    return {
        "total_predictions": total_predictions,
        "avg_confidence": round(avg_confidence, 2),
        "class_distribution": class_distribution,
        "trends": trends
    }

# Initialize database on import
init_db()
