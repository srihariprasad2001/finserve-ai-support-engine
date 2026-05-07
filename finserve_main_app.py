from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import anthropic
import chromadb
import os
import re
import json
import random
from datetime import datetime
from typing import Optional, List, Dict, Any

# =============================================================================
# Section 0 — Initialize FastAPI app and clients
# =============================================================================

app = FastAPI(
    title="FinServe AI Support Engine",
    description="Production AI customer support system combining LLM, RAG, agents, pipelines, and prediction",
    version="2.0.0"
)

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Initialize ChromaDB (in-memory for local, persistent in production)
chroma_client = chromadb.Client()
policy_collection = None  # Initialized on startup

# Global state for request logging
request_log = []

# Global state for tool execution audit trail (in production: use a database)
ticket_log = []
escalation_log = []
acknowledgement_log = []

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
    tool_results: List[Dict[str, Any]]  # NEW: actual execution results
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
    
    # Validate category — fallback to safe default if Claude returns garbage
    valid_categories = ["billing_complaint", "policy_query", "urgent_escalation", "account_request"]
    if category not in valid_categories:
        category = "urgent_escalation"  # Safest default
    
    return category, tokens

# =============================================================================
# Section 5 — Phase 3 Agentic System (REAL TOOL EXECUTION)
# =============================================================================
#
# UPGRADE: These functions now ACTUALLY EXECUTE actions instead of just
# returning tool names. Each function returns structured data that proves
# the action was taken (ticket IDs, escalation IDs, timestamps, etc.).
#
# In production, replace the mock logic with real API calls:
#   - create_refund_ticket → POST to Jira/Zendesk/ServiceNow
#   - escalate_to_agent    → Slack/PagerDuty/Twilio notifications
#   - send_acknowledgement → SendGrid/AWS SES/Mailgun
# =============================================================================

def extract_amount_from_email(email_body: str) -> float:
    """Helper: Extract dollar amount from email body using regex"""
    # Match patterns like $500, $1,500.00, 500 dollars
    pattern = r'\$([\d,]+\.?\d*)'
    matches = re.findall(pattern, email_body)
    if matches:
        # Take the first amount, remove commas, convert to float
        return float(matches[0].replace(',', ''))
    return 0.0

def create_refund_ticket(customer_id: str, amount: float, reason: str) -> dict:
    """
    TOOL: Creates a refund ticket in the ticketing system.
    
    In production, this would POST to Jira/Zendesk/ServiceNow.
    For demo, we generate a mock ticket and log it.
    """
    ticket_id = f"TICKET-{random.randint(10000, 99999)}"
    
    ticket_data = {
        "tool": "create_refund_ticket",
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "type": "refund",
        "amount": amount,
        "reason": reason,
        "status": "pending_investigation",
        "priority": "high",
        "sla_hours": 48,
        "created_at": datetime.now().isoformat(),
        "estimated_resolution": "5-7 business days"
    }
    
    # Log it (in production: save to database)
    ticket_log.append(ticket_data)
    print(f"✓ [EXECUTED] Created refund ticket: {ticket_id} for {customer_id} (${amount})")
    
    # PRODUCTION CODE (commented out — would replace mock logic above):
    # response = requests.post(
    #     "https://api.zendesk.com/v2/tickets.json",
    #     headers={"Authorization": f"Bearer {os.environ.get('ZENDESK_TOKEN')}"},
    #     json={"ticket": {...}}
    # )
    # return response.json()
    
    return ticket_data

def escalate_to_agent(customer_id: str, urgency_level: str, reason: str) -> dict:
    """
    TOOL: Escalates urgent issues to senior support team.
    
    In production, this would send Slack/PagerDuty/Twilio notifications.
    For demo, we generate a mock escalation record.
    """
    escalation_id = f"ESC-{random.randint(10000, 99999)}"
    
    sla_minutes = {"high": 30, "critical": 15}.get(urgency_level.lower(), 60)
    
    escalation_data = {
        "tool": "escalate_to_agent",
        "escalation_id": escalation_id,
        "customer_id": customer_id,
        "urgency": urgency_level,
        "reason": reason,
        "sla_minutes": sla_minutes,
        "assigned_to": "senior-support-team",
        "callback_required": True,
        "created_at": datetime.now().isoformat(),
        "status": "queued_for_immediate_action"
    }
    
    # Log it
    escalation_log.append(escalation_data)
    print(f"✓ [EXECUTED] Escalated to senior team: {escalation_id} (SLA: {sla_minutes} min)")
    
    # PRODUCTION CODE:
    # slack_client.chat_postMessage(
    #     channel="#urgent-support",
    #     text=f"🚨 New escalation: {escalation_id} for {customer_id}"
    # )
    # pagerduty_client.create_incident(...)
    
    return escalation_data

def send_acknowledgement(customer_email: str, customer_id: str, subject: str) -> dict:
    """
    TOOL: Sends acknowledgement email to customer.
    
    In production, this would use SendGrid/AWS SES/Mailgun.
    For demo, we generate a mock send confirmation.
    """
    message_id = f"MSG-{random.randint(100000, 999999)}"
    
    ack_data = {
        "tool": "send_acknowledgement",
        "message_id": message_id,
        "customer_id": customer_id,
        "sent_to": customer_email,
        "subject": f"RE: {subject}",
        "template": "acknowledgement_v1",
        "sent_at": datetime.now().isoformat(),
        "delivery_status": "queued"
    }
    
    # Log it
    acknowledgement_log.append(ack_data)
    print(f"✓ [EXECUTED] Sent acknowledgement to {customer_email}: {message_id}")
    
    # PRODUCTION CODE:
    # sendgrid_client.send(
    #     to=customer_email,
    #     subject="We received your request",
    #     template_id="d-abc123",
    #     dynamic_template_data={"customer_id": customer_id}
    # )
    
    return ack_data

def answer_from_policy(customer_id: str, policy_context: str) -> dict:
    """
    TOOL: Marks that the response is grounded in policy documents.
    
    The actual policy retrieval happens in Phase 2 (RAG) and the
    response generation in Phase 1. This tool just records that
    a policy-based answer was used.
    """
    return {
        "tool": "answer_from_policy",
        "customer_id": customer_id,
        "policy_context_used": bool(policy_context),
        "context_length": len(policy_context) if policy_context else 0,
        "executed_at": datetime.now().isoformat(),
        "status": "policy_grounded_response"
    }

def select_and_execute_tools(email: EmailRequest, category: str, policy_context: str = "") -> dict:
    """
    Phase 3: Select AND EXECUTE appropriate tools based on category.
    
    This is the actual agentic execution layer. Based on the classified
    intent, it triggers the corresponding tool functions and collects
    their execution results for audit trail and API response.
    """
    tools_used = []
    execution_results = []
    
    if category == "billing_complaint":
        # Extract amount from email body for the ticket
        amount = extract_amount_from_email(email.body)
        
        # Tool 1: Create the refund ticket
        ticket_result = create_refund_ticket(
            customer_id=email.customer_id,
            amount=amount,
            reason=f"Customer reported billing issue: {email.subject}"
        )
        tools_used.append("create_refund_ticket")
        execution_results.append(ticket_result)
        
        # Tool 2: Send acknowledgement to customer
        ack_result = send_acknowledgement(
            customer_email=email.email_address,
            customer_id=email.customer_id,
            subject=email.subject
        )
        tools_used.append("send_acknowledgement")
        execution_results.append(ack_result)
        
    elif category == "urgent_escalation":
        # Escalate to senior team immediately
        escalation_result = escalate_to_agent(
            customer_id=email.customer_id,
            urgency_level="high",
            reason=f"Urgent customer issue: {email.subject}"
        )
        tools_used.append("escalate_to_agent")
        execution_results.append(escalation_result)
        
    elif category == "policy_query":
        # Mark policy-grounded response
        policy_result = answer_from_policy(
            customer_id=email.customer_id,
            policy_context=policy_context
        )
        tools_used.append("answer_from_policy")
        execution_results.append(policy_result)
        
    elif category == "account_request":
        # Account requests get acknowledgement + escalation for human review
        ack_result = send_acknowledgement(
            customer_email=email.email_address,
            customer_id=email.customer_id,
            subject=email.subject
        )
        tools_used.append("send_acknowledgement")
        execution_results.append(ack_result)
        
        escalation_result = escalate_to_agent(
            customer_id=email.customer_id,
            urgency_level="medium",
            reason=f"Account change request: {email.subject}"
        )
        tools_used.append("escalate_to_agent")
        execution_results.append(escalation_result)
        
    else:
        # Default fallback: acknowledge receipt
        ack_result = send_acknowledgement(
            customer_email=email.email_address,
            customer_id=email.customer_id,
            subject=email.subject
        )
        tools_used.append("send_acknowledgement")
        execution_results.append(ack_result)
    
    return {
        "tools_used": tools_used,
        "execution_results": execution_results,
        "action_count": len(tools_used)
    }

# =============================================================================
# Section 6 — Phase 1 LLM Integration (Claude)
# =============================================================================

def generate_response(email: EmailRequest, category: str, policy_context: str, tool_results: list) -> tuple:
    """Phase 1: Generate AI response using Claude — now aware of tool results"""
    
    # Build tool execution summary for Claude's context
    tool_summary = ""
    if tool_results:
        actions_taken = []
        for result in tool_results:
            if result.get("tool") == "create_refund_ticket":
                actions_taken.append(f"- Refund ticket {result['ticket_id']} created (resolution in {result['estimated_resolution']})")
            elif result.get("tool") == "escalate_to_agent":
                actions_taken.append(f"- Escalated to senior team (case {result['escalation_id']}, callback within {result['sla_minutes']} minutes)")
            elif result.get("tool") == "send_acknowledgement":
                actions_taken.append(f"- Confirmation email sent (reference {result['message_id']})")
            elif result.get("tool") == "answer_from_policy":
                actions_taken.append("- Response based on FinServe policy documents")
        
        if actions_taken:
            tool_summary = "\n\nActions already taken on this case:\n" + "\n".join(actions_taken)
    
    system_prompt = f"""You are a helpful customer support agent for FinServe AI Support Engine.
Customer ID: {email.customer_id}
Email: {email.email_address}
Category: {category}

If relevant policy information is provided below, use it to answer accurately.
If actions have already been taken, mention them in your response so the customer knows what's happening."""
    
    enriched_prompt = f"""Policy context (if relevant):
{policy_context}
{tool_summary}

Customer subject: {email.subject}
Customer message: {email.body}

Provide a professional, empathetic response that acknowledges any actions already taken."""
    
    response = anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=400,
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
    3. Retrieve policies (Phase 2)
    4. Select AND EXECUTE tools (Phase 3) — now actually creates tickets!
    5. Generate response (Phase 1) — aware of executed actions
    """
    try:
        total_tokens = 0
        
        # Phase 5: Predict urgency
        prediction = predict_urgency(email)
        total_tokens += prediction["ai_tokens"]
        
        # Phase 4: Classify
        category, class_tokens = classify_email(email)
        total_tokens += class_tokens
        
        # Phase 2: Retrieve policies (moved BEFORE Phase 3 so tools can use it)
        policy_context = retrieve_policy_context(email.subject + " " + email.body)
        
        # Phase 3: Select AND EXECUTE tools (now returns real execution results)
        tool_execution = select_and_execute_tools(email, category, policy_context)
        
        # Phase 1: Generate response (now aware of what tools executed)
        response_text, response_tokens = generate_response(
            email, category, policy_context, tool_execution["execution_results"]
        )
        total_tokens += response_tokens
        
        # Log the request
        request_log.append({
            "timestamp": datetime.now().isoformat(),
            "email_id": email.id,
            "customer_id": email.customer_id,
            "urgency": prediction["urgency"],
            "category": category,
            "tools_executed": tool_execution["tools_used"],
            "total_tokens": total_tokens
        })
        
        return FullProcessingResponse(
            email_id=email.id,
            customer_id=email.customer_id,
            urgency_score=prediction["score"],
            urgency_label=prediction["urgency"],
            classification=category,
            tools_used=tool_execution["tools_used"],
            tool_results=tool_execution["execution_results"],  # NEW: actual results
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
        "tools_executed": {
            "tickets_created": len(ticket_log),
            "escalations_created": len(escalation_log),
            "acknowledgements_sent": len(acknowledgement_log)
        },
        "last_request": request_log[-1] if request_log else None
    }

@app.get("/tickets")
async def get_tickets():
    """View all refund tickets created (audit trail)"""
    return {
        "total_tickets": len(ticket_log),
        "tickets": ticket_log
    }

@app.get("/escalations")
async def get_escalations():
    """View all escalations created (audit trail)"""
    return {
        "total_escalations": len(escalation_log),
        "escalations": escalation_log
    }

@app.get("/acknowledgements")
async def get_acknowledgements():
    """View all acknowledgements sent (audit trail)"""
    return {
        "total_acknowledgements": len(acknowledgement_log),
        "acknowledgements": acknowledgement_log
    }

# =============================================================================
# Section 8 — Local Testing (for development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("   FINSERVE AI SUPPORT ENGINE — PHASE 6 (UPGRADED)")
    print("   With REAL Tool Execution")
    print("   Starting FastAPI server...")
    print("=" * 60)
    print("\n   Endpoints:")
    print("   - GET  http://localhost:8000/health")
    print("   - POST http://localhost:8000/predict")
    print("   - POST http://localhost:8000/process")
    print("   - GET  http://localhost:8000/stats")
    print("   - GET  http://localhost:8000/tickets")
    print("   - GET  http://localhost:8000/escalations")
    print("   - GET  http://localhost:8000/acknowledgements")
    print("   - GET  http://localhost:8000/docs (Swagger UI)")
    print("\n   Run with: python finserve_main_app.py\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
