import streamlit as st
import os
from vision import extract_medical_data
from rag_chain import generate_advice

# Ensure standard Streamlit page settings
st.set_page_config(page_title="AI Medical Lab Analyzer", page_icon="🩺", layout="centered")

st.title("🩺 Multimodal AI Lab Report Analyzer")
st.markdown("Upload your blood test report (Image or PDF) and our AI will extract the data and provide personalized dietary advice based on standard medical guidelines.")

# File Uploader
uploaded_file = st.file_uploader("Upload Lab Report", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    # Save the uploaded file temporarily so our scripts can read it
    temp_path = "temp_report." + uploaded_file.name.split('.')[-1]
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    is_pdf = uploaded_file.name.lower().endswith('.pdf')
    if is_pdf:
        st.info(f"📄 PDF Document Uploaded: **{uploaded_file.name}**")
    else:
        st.image(temp_path, caption="Uploaded Report", use_container_width=True)
    
    if st.button("Analyze Report"):
        with st.spinner("Extracting Biomarkers using Vision AI..."):
            try:
                # Run Phase 3: Vision Extraction
                extraction = extract_medical_data(temp_path)
                
                # Display the extracted data in a clean table
                st.subheader("📊 Extracted Results")
                
                # Format into a table
                table_data = [{"Test": item.test_name, "Value": f"{item.value} {item.unit}", "Status": item.status} for item in extraction.results]
                st.table(table_data)
                
                # Run Phase 4: RAG Reasoning
                with st.spinner("Consulting Medical Database..."):
                    try:
                        advice = generate_advice(image_path=temp_path, extraction=extraction)
                    except TypeError:
                        advice = generate_advice(temp_path)
                    
                    st.subheader("💡 Personalized AI Advice")
                    st.info(advice)
                    
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
            finally:
                # Clean up the temporary file safely
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

st.markdown("---")
st.caption("⚠️ **Medical Disclaimer:** This is an experimental AI application. The dietary suggestions are based on general guidelines and should not replace professional medical advice. Always consult a physician.")
