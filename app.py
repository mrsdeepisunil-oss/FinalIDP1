import streamlit as st
import os
import cv2
import numpy as np
import easyocr
import boto3
import psycopg2 
import fitz  # PyMuPDF to process PDF files
from psycopg2.extras import Json
# ==========================================
# 1. ENVIRONMENT CONFIGURATION (SECURED VIA STREAMLIT SECRETS)
# ==========================================
AWS_BUCKET_NAME = st.secrets["AWS_BUCKET_NAME"]
AWS_REGION = st.secrets["AWS_REGION"]
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]

DB_HOST = st.secrets["DB_HOST"]
DB_PORT = int(st.secrets["DB_PORT"])
DB_NAME = st.secrets["DB_NAME"]
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]

# ==========================================
# 2. IMAGE & PDF PREPROCESSING & CLASSIFICATION LOGIC
# ==========================================

def convert_pdf_page_to_bytes(pdf_bytes, page_num=0):
    """Converts a specific PDF page into standard PNG bytes for OpenCV."""
    try:
        # Open the PDF directly from memory
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(page_num)
        
        # Render the page to a high-quality image matrix (pixmap)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Scale up x2 for sharper OCR text
        return pix.tobytes("png")
    except Exception as e:
        st.error(f"Failed to process PDF structure: {str(e)}")
        return None

def clean_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary_cleaned = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return img_rgb, binary_cleaned

def classify_document_rvl_cdip(ocr_results):
    full_text = " ".join([item[1].lower() for item in ocr_results])
    
    if any(k in full_text for k in ["invoice", "total due", "amount", "tax invoice", "subtotal", "qty"]):
        return "Invoice"      # RVL-CDIP Class 15
    elif any(k in full_text for k in ["resume", "cv", "education", "experience", "skills", "projects", "summary"]):
        return "Resume"       # Custom Document Class
    elif any(k in full_text for k in ["identity card", "national id", "dob", "birth date", "issuing authority", "dl no"]):
        return "ID Card"      # RVL-CDIP / Form Class 1
    else:
        return "Letter"       # Default RVL-CDIP Class 0 (Unclassified Memo/Letter)

def store_in_rds_sql(filename, s3_url, structured_payload, avg_confidence):
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, database=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, connect_timeout=10
        )
        cursor = conn.cursor()
        
        insert_query = """
            INSERT INTO compliance_documents (filename, s3_url, status, ocr_confidence_score, extracted_data)
            VALUES (%s, %s, 'COMPLETED', %s, %s)
        """
        cursor.execute(insert_query, (
            filename, s3_url, float(avg_confidence), Json(structured_payload)
        ))
        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

def upload_to_s3(file_bytes, filename, content_type):
    try:
        s3_client = boto3.client(
            's3', aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION
        )
        s3_client.put_object(
            Bucket=AWS_BUCKET_NAME, Key=filename, Body=file_bytes, ContentType=content_type
        )
        s3_url = f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com/{filename}"
        return {"status": "success", "s3_url": s3_url}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==========================================
# 3. INTERACTIVE STREAMLIT INTERFACE
# ==========================================
st.set_page_config(page_title="Intelligent Document Processor", layout="wide")
st.title("🗂️ Intelligent Document Processing System")
st.subheader("Automated Multi-Format (PDF/Image) Classification Pipeline")
st.divider()

# Added "pdf" directly to the allowed formats array
uploaded_file = st.file_uploader("Upload any document (PDF, PNG, or JPG)...", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    processing_bytes = file_bytes  # This holds what OpenCV/EasyOCR will evaluate
    
    with st.spinner("Uploading document to AWS S3 storage core..."):
        cloud_response = upload_to_s3(file_bytes, uploaded_file.name, uploaded_file.type)
        
    if cloud_response["status"] == "success":
        st.success("✅ Original file safely mirrored to AWS S3 Cloud!")
        s3_url = cloud_response["s3_url"]
        
        # 📂 ROUTING CRITICAL STEP: Convert to image if it's a PDF file
        if uploaded_file.type == "application/pdf":
            with st.spinner("Rendering vector PDF page structure into OCR image layers..."):
                processing_bytes = convert_pdf_page_to_bytes(file_bytes, page_num=0)
        
        if processing_bytes is not None:
            with st.spinner("Applying OpenCV pixel optimization layers..."):
                img_rgb, binary_cleaned = clean_image(processing_bytes)
                
            if binary_cleaned is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🖼️ Preprocessing View")
                    # 👇 ADD THIS LINE BELOW TO SHOW THE COLOR IMAGE
                    st.image(img_rgb, caption="Original Color Document", use_container_width=True)
                    
                    # This is your existing black and white image display line
                    st.image(binary_cleaned, caption="Binarized Matrix Target (Page 1)", use_container_width=True, channels="GRAY")
                
                with col2:
                    st.markdown("### 💾 AI Intelligent Extraction Ledger")
                    
                    with st.spinner("Extracting text and layout elements..."):
                        reader = load_ocr_reader()
                        ocr_results = reader.readtext(binary_cleaned)
                    
                    if ocr_results:
                        raw_text_list = []
                        confidence_scores = []
                        bounding_boxes = []

                        for idx, (bbox, text, conf) in enumerate(ocr_results):
                            raw_text_list.append(text)
                            confidence_scores.append(conf)
                            clean_bbox = [[int(coord[0]), int(coord[1])] for coord in bbox]
                            bounding_boxes.append({f"element_{idx}": clean_bbox})

                        combined_text = "\n".join(raw_text_list)
                        avg_confidence = np.mean(confidence_scores) * 100 if confidence_scores else 0.0
                        
                        with st.spinner("Running document layout classification..."):
                            predicted_class = classify_document_rvl_cdip(ocr_results)
                        
                        st.info(f"🔮 **Predicted Document Type:** {predicted_class}")
                        
                        structured_payload = {
                            "predicted_class_type": predicted_class,
                            "spatial_bounding_boxes": bounding_boxes,
                            "raw_ocr_dump": combined_text
                        }

                        st.metric(label="OCR Average Extraction Confidence Score", value=f"{avg_confidence:.2f}%")
                        st.json(structured_payload)
                        
                        with st.spinner("Writing records into your AWS RDS database instance..."):
                            db_response = store_in_rds_sql(
                                uploaded_file.name, s3_url, structured_payload, avg_confidence
                            )
                            
                        if db_response["status"] == "success":
                            st.success(f"💾 Structured metadata committed under category: {predicted_class}!")
                        else:
                            st.error(f"RDS SQL Write Exception: {db_response['message']}")
                    else:
                        st.warning("OCR complete, but could not read text blocks.")
        else:
            st.error("Could not extract image arrays from the uploaded PDF document.")
    else:
        st.error(f"S3 Storage abort: {cloud_response['message']}")