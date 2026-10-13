import os
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64
from io import BytesIO

# Load environment variables (your API key)
load_dotenv()

# 1. Define the exact JSON structure we want Gemini to return
class Biomarker(BaseModel):
    test_name: str = Field(description="Name of the test, e.g., Hemoglobin, LDL Cholesterol")
    value: str = Field(description="The numeric value extracted")
    unit: str = Field(description="The unit of measurement, e.g., g/dL, mg/dL")
    status: str = Field(description="Flag or status, e.g., Normal, High, Low")

class BloodTestReport(BaseModel):
    results: List[Biomarker]

def process_file(file_path: str) -> List[Image.Image]:
    """Converts a PDF to a list of Images, or just opens an Image directly."""
    images = []
    if file_path.lower().endswith('.pdf'):
        print("PDF detected. Converting pages to images...")
        with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
    else:
        print("Image detected.")
        with Image.open(file_path) as img:
            images.append(img.copy())
    return images

def image_to_base64(img: Image.Image) -> str:
    """Helper function to convert a PIL Image to a base64 string"""
    buffered = BytesIO()
    # Convert to RGB in case it's a PNG with transparency
    img.convert('RGB').save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_medical_data(file_path: str):
    print(f"\nExtracting data from {file_path}...")
    images = process_file(file_path)
    
    # 2. Initialize the Vision LLM
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    
    # 3. Force the LLM to return our exact JSON structure
    structured_llm = llm.with_structured_output(BloodTestReport)
    
    prompt = """
    You are an expert medical data extractor. 
    Analyze this blood test report image and extract all the biomarker readings.
    Ensure you capture the test name, value, unit, and any high/low flags.
    Ignore patient personal details like name and address.
    """
    
    all_results = []
    
    # Process every page in the document
    print("Sending to Gemini Vision...")
    for i, img in enumerate(images):
        print(f"Processing page {i+1} of {len(images)}...")
        
        # 4. Convert Image to base64 and construct the multimodal message
        base64_img = image_to_base64(img)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]
        )
        
        response = structured_llm.invoke([message])
        if response and response.results:
            all_results.extend(response.results)
            
    # Combine all extracted biomarkers into a single report
    return BloodTestReport(results=all_results)

if __name__ == "__main__":
    # To test this, you will need to put a sample blood test image 
    # (like "sample.jpg") in your folder and uncomment the lines below.
    
    results = extract_medical_data("sample.png")
    print("\n--- EXTRACTION RESULTS ---")
    for item in results.results:
        print(f"{item.test_name}: {item.value} {item.unit} ({item.status})")
    pass
