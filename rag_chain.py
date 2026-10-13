import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vision import extract_medical_data

def generate_advice(image_path: str = None, extraction = None):
    # 1. Extract the data using our vision script if not already provided
    if extraction is None:
        if not image_path:
            raise ValueError("Either image_path or extraction must be provided.")
        extraction = extract_medical_data(image_path)
    
    # 2. Filter out normal results to focus only on Abnormalities (High/Low)
    abnormal_results = [
        item for item in extraction.results 
        if item.status.lower() in ["high", "low"]
    ]
    
    if not abnormal_results:
        return "Great news! All your biomarkers are within the normal range. Keep maintaining a healthy lifestyle."
    
    # Format the abnormal findings for our search
    findings_str = ", ".join([f"{item.test_name} is {item.status} ({item.value} {item.unit})" for item in abnormal_results])
    print(f"\n[!] Identified Abnormalities: {findings_str}")
    
    # 3. Connect to our existing Chroma Database
    print("\n[!] Searching Medical Database for advice...")
    if not os.path.exists("./chroma_db"):
        print("[!] chroma_db not found. Auto-ingesting reference data...")
        from ingest import ingest_data
        ingest_data()
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # Search the database for the 3 most relevant paragraphs
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(findings_str)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 4. Initialize our Reasoning LLM
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
    
    # 5. Create the Prompt Template combining User Data + Retrieved Database Rules
    prompt = ChatPromptTemplate.from_template("""
    You are an expert AI Medical and Nutritional Assistant.
    A user has uploaded a blood test with the following abnormal results:
    {findings}
    
    Here is the medical reference data and dietary suggestions retrieved from our database:
    {context}
    
    Task:
    1. Briefly explain what these abnormal results might mean.
    2. Provide the actionable dietary and lifestyle suggestions based STRICTLY on the retrieved database context.
    3. End with a strict medical disclaimer that this is an AI tool and they should consult a doctor.
    """)
    
    # 6. Generate the final personalized report
    print("\n[!] Generating your personalized health report...\n")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "findings": findings_str,
        "context": context
    })
    
    return response

if __name__ == "__main__":
    advice = generate_advice("sample.png")
    print("================ YOUR PERSONALIZED ADVICE ================\n")
    print(advice)
