"""
FinServe AI Support Engine — Phase 6
Complete Production Application

Combines all 5 phases into one deployable FastAPI system:
  - Phase 1: LLM API (Claude integration)
  - Phase 2: RAG (vector database with policies)
  - Phase 3: Agentic workflows (tool selection and execution)
  - Phase 4: Data pipelines (batch processing)
  - Phase 5: Predictive backend (urgency scoring)

Deployment targets:
  - Local: python finserve_main_app.py
  - GCP Cloud Run: gcloud run deploy finserve-api --source . --platform managed --region us-central1

Environment variables required:
  - ANTHROPIC_API_KEY: Your Claude API key
  - DATABASE_URL: PostgreSQL connection string (optional, for production)
  - GCP_PROJECT: Your GCP project ID (optional, for production)
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import anthropic
import chromadb
import os
import re
import json
from datetime import datetime
from typing import Optional

# =============================================================================
# Section 0 — Initialize FastAPI app and clients
# =============================================================================

app = FastAPI(
    title="FinServe AI Support Engine",
    description="Production AI customer support system combining LLM, RAG, agents, pipelines, and prediction",
    version="1.0.0"
)

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Initialize ChromaDB (in-memory for local, persistent in production)
chroma_client = chromadb.Client()
policy_collection = None  # Initialized on startup

# Global state for request logging
request_log = []

# =============================================================================
# Section 1 — Data Models (Pydantic)
# =============================================================================

class EmailRequest(BaseModel):
    """Input model for email processing"""
    id: str
    customer_id: str
    email_address: str
    subject: str
    body: str
    hour_sent: int = 12
    prior_complaints: int = 0

class PredictionResponse(BaseModel):
    """Response model for urgency prediction"""
    email_id: str
    score: int
    urgency: str
    ai_used: bool
    tokens_used: int
    timestamp: str

class FullProcessingResponse(BaseModel):
    """Response model for full pipeline processing"""
    email_id: str
    customer_id: str
    urgency_score: int
    urgency_label: str
    classification: str
    tools_used: list
    claude_response: str
    total_tokens_used: int
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    rag_ready: bool
    anthropic_ready: bool

# =============================================================================
# Section 2 — Phase 2 RAG System (Policies)
# =============================================================================

def initialize_rag():
    """Initialize RAG system with policy documents"""
    global policy_collection
    
    policy_chunks = [
        {
            "id": "policy_001",
            "text": "FinServe Credit late payment policy: A late payment fee of $25 is charged when a payment is received more than 5 business days after the due date. Customers who have never missed a payment before may request a one-time waiver by contacting support."
        },
        {
            "id": "policy_002",
            "text": "FinServe Credit account closure policy: Customers may close their account at any time by submitting a written request. All outstanding balances must be paid in full before closure. Account closure typically takes 5-7 business days to process."
        },
        {
            "id": "policy_003",
            "text": "FinServe Credit loan balance enquiry: Customers can check their outstanding loan balance by logging into the FinServe mobile app, calling the 24/7 helpline at 1800-XXX-XXXX, or visiting any branch with a valid photo ID."
        },
        {
            "id": "policy_004",
            "text": "FinServe Credit duplicate charge policy: If a customer reports a duplicate payment deduction, the support team will investigate within 2 business days. Confirmed duplicates are refunded within 5-7 business days to the original payment method."
        },
        {
            "id": "policy_005",
            "text": "FinServe Credit interest rate policy: Personal loan interest rates range from 8.5% to 18% per annum based on credit score and loan tenure. Customers can request a rate review after 12 months of consistent on-time payments."
        },
        {
            "id": "policy_006",
            "text": "FinServe Credit dispute resolution: Customers who wish to dispute a charge or decision may submit a formal complaint via email at disputes@finservecredit.com. All disputes are reviewed within 3 business days and customers are notified of the outcome."
        }
    ]
    
    policy_collection = chroma_client.create_collection(name="finserve_policies")
    policy_collection.add(
        documents=[chunk["text"] for chunk in policy_chunks],
        ids=[chunk["id"] for chunk in policy_chunks]
    )
    return True

def retrieve_policy_context(question: str) -> str:
    """Phase 2: Retrieve relevant policies for a question"""
    if policy_collection is None:
        return ""
    
    results = policy_collection.query(
        query_texts=[question],
        n_results=2
    )
    
    if results and results["documents"]:
        return "\n\n".join(results["documents"][0])
    return ""

# =============================================================================
# Section 3 — Phase 5 Predictive System
# =============================================================================

def extract_features(email: EmailRequest) -> dict:
    """Phase 5: Extract features from email"""
    body = email.body.lower()
    subject = email.subject.lower()
    full_text = subject + " " + body
    
    exclamation_count = full_text.count("!")
    urgent_keywords = ["urgent", "furious", "unacceptable", "regulator", "dispute", "third time", "immediately", "frustrated", "angry", "outraged"]
    keyword_hits = sum(1 for word in urgent_keywords if word in full_text)
    word_count = len(full_text.split())
    off_hours = email.hour_sent < 7 or email.hour_sent > 21
    caps_words = len(re.findall(r'\b[A-Z]{3,}\b', email.subject + " " + email.body))
    
    return {
        "exclamation_count": exclamation_count,
        "keyword_hits": keyword_hits,
        "word_count": word_count,
        "off_hours": off_hours,
        "prior_complaints": email.prior_complaints,
        "caps_words": caps_words
    }

def calculate_urgency_score(features: dict) -> int:
    """Phase 5: Calculate urgency score"""
    score = 0
    score += min(features["exclamation_count"] * 4, 20)
    score += min(features["keyword_hits"] * 10, 30)
    if features["off_hours"]:
        score += 15
    score += min(features["prior_complaints"] * 5, 20)
    score += min(features["caps_words"] * 3, 10)
    if features["word_count"] > 80:
        score += 5
    return min(score, 100)

def ai_confirm_urgency(email: EmailRequest, score: int) -> tuple:
    """Phase 5: Use Claude to confirm borderline urgency"""
    prompt = f"""You are an urgency classifier for FinServe AI Support Engine.
A rule-based system scored this email {score}/100.
Confirm or adjust: reply with ONLY one word: LOW, MEDIUM, or HIGH.
Subject: {email.subject}
Body: {email.body}"""
    
    response = anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    
    ai_label = response.content[0].text.strip().upper()
    if ai_label not in ["LOW", "MEDIUM", "HIGH"]:
        ai_label = "MEDIUM"
    
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return ai_label, tokens

def predict_urgency(email: EmailRequest) -> dict:
    """Phase 5: Full prediction pipeline"""
    features = extract_features(email)
    score = calculate_urgency_score(features)
    ai_tokens = 0
    ai_used = False
    
    if score >= 71:
        urgency = "HIGH"
    elif score >= 40:
        urgency, ai_tokens = ai_confirm_urgency(email, score)
        ai_used = True
    else:
        urgency = "LOW"
    
    return {
        "score": score,
        "urgency": urgency,
        "ai_used": ai_used,
        "ai_tokens": ai_tokens
    }

# =============================================================================
# Section 4 — Phase 4 Classification System
# =============================================================================

def classify_email(email: EmailRequest) -> tuple:
    """Phase 4: Classify email intent"""
    prompt = f"""Classify this email into ONE category:
- billing_complaint: duplicate charges, wrong amounts, payment errors
- policy_query: questions about fees, rates, balances, processes
- urgent_escalation: angry customers, unresolved issues, escalation requests
- account_request: account closure, updates, changes

Subject: {email.subject}
Body: {email.body}

Reply with ONLY the category name."""
    
    response = anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}]
    )
    
    category = response.content[0].text.strip().lower()
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return category, tokens

# =============================================================================
# Section 5 — Phase 3 Agentic System (Tool Selection)
# =============================================================================

def select_and_execute_tools(email: EmailRequest, category: str) -> dict:
    """Phase 3: Select appropriate tools based on category"""
    tools_used = []
    
    if category == "billing_complaint":
        tools_used.append("create_refund_ticket")
        tools_used.append("send_acknowledgement")
    elif category == "urgent_escalation":
        tools_used.append("escalate_to_agent")
    elif category == "policy_query":
        tools_used.append("answer_from_policy")
    else:
        tools_used.append("send_acknowledgement")
    
    return {"tools_used": tools_used, "action_count": len(tools_used)}

# =============================================================================
# Section 6 — Phase 1 LLM Integration (Claude)
# =============================================================================

def generate_response(email: EmailRequest, category: str, policy_context: str) -> tuple:
    """Phase 1: Generate AI response using Claude"""
    system_prompt = f"""You are a helpful customer support agent for FinServe AI Support Engine.
Customer ID: {email.customer_id}
Email: {email.email_address}
Category: {category}

If relevant policy information is provided below, use it to answer accurately."""
    
    enriched_prompt = f"""Policy context (if relevant):
{policy_context}

Customer subject: {email.subject}
Customer message: {email.body}

Provide a professional, empathetic response."""
    
    response = anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": enriched_prompt}]
    )
    
    answer = response.content[0].text
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return answer, tokens

# =============================================================================
# Section 7 — API Endpoints
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    try:
        initialize_rag()
    except Exception as e:
        print(f"Warning: RAG initialization failed: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        rag_ready=policy_collection is not None,
        anthropic_ready=anthropic_client is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_only(email: EmailRequest):
    """
    Phase 5 only: Predict urgency without full processing
    Useful for quick routing decisions before detailed processing
    """
    try:
        prediction = predict_urgency(email)
        
        return PredictionResponse(
            email_id=email.id,
            score=prediction["score"],
            urgency=prediction["urgency"],
            ai_used=prediction["ai_used"],
            tokens_used=prediction["ai_tokens"],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=FullProcessingResponse)
async def process_email(email: EmailRequest):
    """
    Full pipeline: Combines all 5 phases
    1. Predict urgency (Phase 5)
    2. Classify intent (Phase 4)
    3. Select tools (Phase 3)
    4. Retrieve policies (Phase 2)
    5. Generate response (Phase 1)
    """
    try:
        total_tokens = 0
        
        # Phase 5: Predict urgency
        prediction = predict_urgency(email)
        total_tokens += prediction["ai_tokens"]
        
        # Phase 4: Classify
        category, class_tokens = classify_email(email)
        total_tokens += class_tokens
        
        # Phase 3: Select tools
        tool_selection = select_and_execute_tools(email, category)
        
        # Phase 2: Retrieve policies
        policy_context = retrieve_policy_context(email.subject + " " + email.body)
        
        # Phase 1: Generate response
        response_text, response_tokens = generate_response(email, category, policy_context)
        total_tokens += response_tokens
        
        # Log the request
        request_log.append({
            "timestamp": datetime.now().isoformat(),
            "email_id": email.id,
            "customer_id": email.customer_id,
            "urgency": prediction["urgency"],
            "category": category,
            "total_tokens": total_tokens
        })
        
        return FullProcessingResponse(
            email_id=email.id,
            customer_id=email.customer_id,
            urgency_score=prediction["score"],
            urgency_label=prediction["urgency"],
            classification=category,
            tools_used=tool_selection["tools_used"],
            claude_response=response_text,
            total_tokens_used=total_tokens,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Analytics endpoint: Returns processing statistics"""
    if not request_log:
        return {"message": "No requests processed yet"}
    
    total_requests = len(request_log)
    total_tokens = sum(r["total_tokens"] for r in request_log)
    urgency_counts = {}
    
    for req in request_log:
        urgency = req["urgency"]
        urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
    
    return {
        "total_requests_processed": total_requests,
        "total_tokens_used": total_tokens,
        "avg_tokens_per_request": total_tokens // total_requests if total_requests > 0 else 0,
        "urgency_breakdown": urgency_counts,
        "last_request": request_log[-1] if request_log else None
    }

@app.get("/docs")
async def docs():
    """Swagger UI documentation — automatically generated by FastAPI"""
    pass  # FastAPI handles this automatically

# =============================================================================
# Section 8 — Local Testing (for development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("   FINSERVE AI SUPPORT ENGINE — PHASE 6")
    print("   Starting FastAPI server...")
    print("=" * 60)
    print("\n   Endpoints:")
    print("   - GET  http://localhost:8000/health")
    print("   - POST http://localhost:8000/predict")
    print("   - POST http://localhost:8000/process")
    print("   - GET  http://localhost:8000/stats")
    print("   - GET  http://localhost:8000/docs (Swagger UI)")
    print("\n   Run with: python finserve_main_app.py\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
