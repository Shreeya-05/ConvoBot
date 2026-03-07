"""
ConvoBot — All-in-One Launcher
===============================
Run this single file to start the entire app:

    python3 run.py

Then open: http://localhost:8001
No separate HTML file needed. Everything is served by FastAPI.

Stack : FastAPI + Groq API (Llama 3.3 70B)
Author: Your Name
"""

import os
import uvicorn
import httpx
import logging
import json
import re
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class Speaker(str, Enum):
    CUSTOMER = "Customer"
    AGENT    = "Agent"

class TranscriptSegment(BaseModel):
    speaker: Speaker
    text:    str = Field(..., min_length=1)

class AnalyticsRequest(BaseModel):
    segments:   List[TranscriptSegment]
    session_id: Optional[str] = None

class AnalyticsResponse(BaseModel):
    intent:             str
    intent_confidence:  int
    topic:              str
    topic_confidence:   int
    sentiment:          str
    sentiment_confidence: int
    escalation_risk:    str
    escalation_signals: List[str] = []
    summary:            str = ""
    key_phrases:        List[str] = []
    segment_count:      int
    session_id:         Optional[str] = None

class ChatMessage(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    messages:   List[ChatMessage]
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply:      str
    session_id: Optional[str] = None

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────
ANALYTICS_SYSTEM = """You are a real-time conversation analytics engine for a call center.
Analyze the provided conversation transcript and return ONLY a valid JSON object:
{
  "intent": "<billing dispute|account inquiry|payment issue|technical support|service complaint|subscription cancellation>",
  "intent_confidence": <0-100>,
  "topic": "<credit card charges|account access issues|subscription cancellation|service outage|duplicate transaction|password reset|other>",
  "topic_confidence": <0-100>,
  "sentiment": "<positive|neutral|negative>",
  "sentiment_confidence": <0-100>,
  "escalation_risk": "<high|medium|low>",
  "escalation_signals": ["exact phrase from transcript"],
  "summary": "one sentence summary",
  "key_phrases": ["phrase1","phrase2","phrase3"]
}
Return ONLY JSON. No markdown, no explanation."""

CHAT_SYSTEM = """You are a helpful, professional customer support agent.
Keep responses concise (2-3 sentences). Be empathetic and solution-focused.
You handle billing, account issues, technical problems, and subscriptions."""

# ─────────────────────────────────────────────
# GROQ HELPER
# ─────────────────────────────────────────────
async def groq_call(messages: list, api_key: str, json_mode: bool = False) -> str:
    key = api_key or GROQ_API_KEY
    if not key:
        raise ValueError("No Groq API key provided.")

    body = {
        "model":      GROQ_MODEL,
        "max_tokens": 1024,
        "temperature": 0.1 if json_mode else 0.7,
        "messages":   messages,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
        )

    data = res.json()
    if res.status_code != 200:
        raise ValueError(data.get("error", {}).get("message", "Groq API error"))
    return data["choices"][0]["message"]["content"]

# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(title="ConvoBot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, x_groq_api_key: Optional[str] = Header(None)):
    """Get an AI bot reply for the conversation."""
    messages = [{"role": "system", "content": CHAT_SYSTEM}]
    for m in request.messages:
        messages.append({"role": m.role, "content": m.content})
    try:
        reply = await groq_call(messages, x_groq_api_key or "", json_mode=False)
        return ChatResponse(reply=reply, session_id=request.session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze", response_model=AnalyticsResponse)
async def analyze(request: AnalyticsRequest, x_groq_api_key: Optional[str] = Header(None)):
    """Analyze transcript for intent, sentiment, escalation risk."""
    transcript = "\n".join(f"{s.speaker}: {s.text}" for s in request.segments)
    try:
        raw = await groq_call(
            messages=[
                {"role": "system", "content": ANALYTICS_SYSTEM},
                {"role": "user",   "content": f"Analyze:\n\n{transcript}"},
            ],
            api_key=x_groq_api_key or "",
            json_mode=True,
        )
        cleaned = re.sub(r"```json|```", "", raw).strip()
        parsed  = json.loads(cleaned)
        return AnalyticsResponse(
            intent=parsed.get("intent", "service complaint"),
            intent_confidence=int(parsed.get("intent_confidence", 0)),
            topic=parsed.get("topic", "other"),
            topic_confidence=int(parsed.get("topic_confidence", 0)),
            sentiment=parsed.get("sentiment", "neutral"),
            sentiment_confidence=int(parsed.get("sentiment_confidence", 0)),
            escalation_risk=parsed.get("escalation_risk", "low"),
            escalation_signals=parsed.get("escalation_signals", []),
            summary=parsed.get("summary", ""),
            key_phrases=parsed.get("key_phrases", []),
            segment_count=len(request.segments),
            session_id=request.session_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# FRONTEND — served directly by Python
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def frontend():
    """Serves the entire chat UI from Python — no separate HTML file needed."""
    return HTMLResponse(content=HTML_PAGE)


# ─────────────────────────────────────────────
# EMBEDDED HTML (entire frontend in one string)
# ─────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ConvoBot — Real-Time Analytics</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');
    *{box-sizing:border-box;margin:0;padding:0}
    :root{--bg:#0A0E1A;--surface:#111827;--surface2:#1C2333;--border:#1F2D45;--accent:#00D9FF;--green:#00FF94;--red:#FF3B6B;--yellow:#FFD166;--purple:#A78BFA;--text:#E2E8F0;--muted:#64748B;--dim:#334155}
    body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
    header{background:var(--surface);border-bottom:1px solid var(--border);padding:14px 24px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
    .logo{font-family:'IBM Plex Mono',monospace;font-size:16px;font-weight:700;color:var(--accent);letter-spacing:2px;display:flex;align-items:center;gap:10px}
    .live-dot{width:8px;height:8px;border-radius:50%;background:var(--muted);transition:background .3s}
    .live-dot.active{background:var(--green);animation:blink 1s infinite}
    .server-status{font-family:'IBM Plex Mono',monospace;font-size:10px;padding:5px 10px;border-radius:6px;border:1px solid var(--border);color:var(--muted);background:var(--surface2);white-space:nowrap}
    .server-status.ok{color:var(--green);border-color:rgba(0,255,148,.3)}
    .server-status.err{color:var(--red);border-color:rgba(255,59,107,.3)}
    .main{display:grid;grid-template-columns:1fr 320px;flex:1;overflow:hidden}
    .chat-panel{display:flex;flex-direction:column;border-right:1px solid var(--border);overflow:hidden}
    .messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
    .messages::-webkit-scrollbar{width:4px}
    .messages::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
    .msg{display:flex;gap:10px;align-items:flex-start;animation:fadeUp .3s ease}
    .msg.customer{justify-content:flex-start}
    .msg.bot{justify-content:flex-end}
    .avatar{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;flex-shrink:0}
    .avatar.c{background:linear-gradient(135deg,#FF3B6B88,#A78BFA88)}
    .avatar.b{background:linear-gradient(135deg,#A78BFA88,#00D9FF88)}
    .bubble-wrap{max-width:70%}
    .bubble-label{font-size:10px;color:var(--muted);margin-bottom:3px;letter-spacing:1px;font-family:'IBM Plex Mono',monospace}
    .msg.bot .bubble-label{text-align:right}
    .bubble{padding:10px 14px;font-size:13px;line-height:1.6;color:var(--text)}
    .msg.customer .bubble{background:var(--surface2);border:1px solid var(--border);border-radius:4px 12px 12px 12px}
    .msg.bot .bubble{background:rgba(167,139,250,.08);border:1px solid rgba(167,139,250,.2);border-radius:12px 4px 12px 12px}
    .typing-indicator{display:none;gap:5px;padding:10px 14px;background:var(--surface2);border:1px solid var(--border);border-radius:4px 12px 12px 12px;width:fit-content}
    .typing-indicator.show{display:flex}
    .typing-indicator span{width:6px;height:6px;border-radius:50%;background:var(--muted);animation:blink 1s infinite}
    .typing-indicator span:nth-child(2){animation-delay:.2s}
    .typing-indicator span:nth-child(3){animation-delay:.4s}
    .input-bar{padding:16px 20px;border-top:1px solid var(--border);background:var(--surface);display:flex;gap:10px;align-items:center;flex-shrink:0}
    .msg-input{flex:1;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:10px 14px;color:var(--text);font-size:13px;font-family:inherit;outline:none;transition:border-color .2s}
    .msg-input:focus{border-color:var(--accent)}
    .send-btn{background:var(--accent);color:var(--bg);border:none;border-radius:8px;padding:10px 20px;font-weight:700;font-size:13px;cursor:pointer;font-family:inherit}
    .send-btn:disabled{opacity:.4;cursor:not-allowed}
    .clear-btn{background:transparent;color:var(--muted);border:1px solid var(--border);border-radius:8px;padding:10px 14px;cursor:pointer;font-size:12px;font-family:inherit}
    .analytics-panel{overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
    .analytics-panel::-webkit-scrollbar{width:4px}
    .analytics-panel::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
    .panel-title{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:2px;color:var(--muted);text-transform:uppercase;padding-bottom:8px;border-bottom:1px solid var(--border)}
    .card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;transition:border-color .3s,box-shadow .3s}
    .card.highlight{border-color:rgba(0,217,255,.4)}
    .card.danger{border-color:rgba(255,59,107,.5);box-shadow:0 0 16px rgba(255,59,107,.1)}
    .card-label{font-size:9px;font-family:'IBM Plex Mono',monospace;letter-spacing:2px;color:var(--muted);text-transform:uppercase;margin-bottom:8px}
    .card-value{font-size:14px;font-weight:700;text-transform:capitalize;color:var(--text)}
    .conf-bar-bg{height:3px;background:var(--border);border-radius:3px;overflow:hidden;margin-top:6px}
    .conf-bar-fill{height:100%;border-radius:3px;transition:width .8s ease}
    .conf-label{font-size:9px;color:var(--muted);text-align:right;margin-top:2px;font-family:'IBM Plex Mono',monospace}
    .risk-badge{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:6px;font-size:12px;font-weight:700;letter-spacing:1px;font-family:'IBM Plex Mono',monospace}
    .risk-dot{width:8px;height:8px;border-radius:50%}
    .signal-item{font-size:11px;color:var(--muted);border-left:2px solid rgba(255,59,107,.4);padding-left:8px;margin-top:6px;line-height:1.5}
    .phrase-wrap{display:flex;flex-wrap:wrap;gap:5px;margin-top:4px}
    .phrase-tag{background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:2px 8px;font-size:10px;color:var(--muted)}
    .empty-state{text-align:center;color:var(--dim);font-size:12px;margin-top:20px;line-height:1.8;font-family:'IBM Plex Mono',monospace}
    .json-box{background:#0D1117;border:1px solid var(--border);border-radius:8px;padding:12px;font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--muted);white-space:pre-wrap;word-break:break-word;line-height:1.7;max-height:180px;overflow-y:auto}
    @keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
    @keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
    @keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(255,59,107,.6)}50%{box-shadow:0 0 0 6px rgba(255,59,107,0)}}
  </style>
</head>
<body>
<header>
  <div class="logo">
    <div class="live-dot active" id="liveDot"></div>
    CONVOBOT
    <span style="font-size:10px;color:var(--muted);font-weight:400">// PYTHON + FASTAPI + GROQ</span>
  </div>
</header>
<div class="main">
  <div class="chat-panel">
    <div class="messages" id="messages">
      <div class="msg bot">
        <div class="bubble-wrap">
          <div class="bubble-label">BOT</div>
          <div class="bubble">👋 Hello! I'm your AI support assistant. How can I help you today?</div>
        </div>
        <div class="avatar b">B</div>
      </div>
    </div>
    <div style="padding:0 20px 8px">
      <div class="typing-indicator" id="typingIndicator"><span></span><span></span><span></span></div>
    </div>
    <div class="input-bar">
      <input class="msg-input" id="msgInput" type="text" placeholder="Type your message and press Enter..."
        onkeydown="if(event.key==='Enter')sendMessage()"/>
      <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
      <button class="clear-btn" onclick="clearChat()">Clear</button>
    </div>
  </div>
  <div class="analytics-panel">
    <div class="panel-title">Live Analytics</div>
    <div id="emptyState" class="empty-state">Start chatting to see<br/>real-time analytics here.</div>
    <div id="analyticsCards" style="display:none;flex-direction:column;gap:10px">
      <div class="card" id="cardIntent">
        <div class="card-label">Intent</div>
        <div class="card-value" id="intentVal">–</div>
        <div class="conf-bar-bg"><div class="conf-bar-fill" id="intentBar" style="width:0%;background:var(--accent)"></div></div>
        <div class="conf-label" id="intentConf">–</div>
      </div>
      <div class="card">
        <div class="card-label">Topic</div>
        <div class="card-value" id="topicVal" style="color:var(--purple)">–</div>
        <div class="conf-bar-bg"><div class="conf-bar-fill" id="topicBar" style="width:0%;background:var(--purple)"></div></div>
        <div class="conf-label" id="topicConf">–</div>
      </div>
      <div class="card">
        <div class="card-label">Sentiment</div>
        <div style="display:flex;align-items:center;gap:6px">
          <span id="sentimentIcon" style="font-size:20px">–</span>
          <span class="card-value" id="sentimentVal">–</span>
        </div>
        <div class="conf-bar-bg"><div class="conf-bar-fill" id="sentimentBar" style="width:0%"></div></div>
        <div class="conf-label" id="sentimentConf">–</div>
      </div>
      <div class="card" id="cardRisk">
        <div class="card-label">Escalation Risk</div>
        <div id="riskBadge" class="risk-badge">
          <div class="risk-dot" id="riskDot"></div>
          <span id="riskVal">–</span>
        </div>
        <div id="signalsList"></div>
      </div>
      <div class="card">
        <div class="card-label">Summary</div>
        <div id="summaryVal" style="font-size:12px;color:var(--muted);line-height:1.6">–</div>
      </div>
      <div class="card">
        <div class="card-label">Key Phrases</div>
        <div class="phrase-wrap" id="phrasesWrap"></div>
      </div>
      <div>
        <div class="panel-title" style="margin-bottom:8px">JSON Output</div>
        <div class="json-box" id="jsonBox">// awaiting input...</div>
      </div>
    </div>
  </div>
</div>
<script>
  let chatHistory=[],segments=[],isBusy=false;
  function addMessage(spk,text){
    const c=document.getElementById('messages'),isBot=spk!=='Customer';
    c.insertAdjacentHTML('beforeend',`<div class="msg ${isBot?'bot':'customer'}">${!isBot?'<div class="avatar c">C</div>':''}<div class="bubble-wrap"><div class="bubble-label">${spk.toUpperCase()}</div><div class="bubble">${text}</div></div>${isBot?'<div class="avatar b">B</div>':''}</div>`);
    c.scrollTop=c.scrollHeight;
  }
  function showTyping(show){
    document.getElementById('typingIndicator').classList.toggle('show',show);
    document.getElementById('messages').scrollTop=99999;
  }
  async function apiFetch(path,body){
    const res=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(!res.ok){const e=await res.json();throw new Error(e.detail||'API error')}
    return res.json();
  }
  async function sendMessage(){
    const input=document.getElementById('msgInput');
    const text=input.value.trim();
    if(!text||isBusy)return;
    input.value='';isBusy=true;
    document.getElementById('sendBtn').disabled=true;
    addMessage('Customer',text);
    segments.push({speaker:'Customer',text});
    chatHistory.push({role:'user',content:text});
    showTyping(true);
    try{
      const data=await apiFetch('/api/chat',{messages:chatHistory,session_id:'web'});
      showTyping(false);
      addMessage('Bot',data.reply);
      segments.push({speaker:'Agent',text:data.reply});
      chatHistory.push({role:'assistant',content:data.reply});
    }catch(e){showTyping(false);addMessage('Bot',`⚠ ${e.message}`);}
    try{const a=await apiFetch('/api/analyze',{segments,session_id:'web'});if(a)updateUI(a);}
    catch(e){console.error(e)}
    isBusy=false;
    document.getElementById('sendBtn').disabled=false;
    input.focus();
  }
  function updateUI(a){
    document.getElementById('emptyState').style.display='none';
    document.getElementById('analyticsCards').style.display='flex';
    document.getElementById('intentVal').textContent=a.intent||'–';
    document.getElementById('intentBar').style.width=(a.intent_confidence||0)+'%';
    document.getElementById('intentConf').textContent=(a.intent_confidence||0)+'% confidence';
    document.getElementById('topicVal').textContent=a.topic||'–';
    document.getElementById('topicBar').style.width=(a.topic_confidence||0)+'%';
    document.getElementById('topicConf').textContent=(a.topic_confidence||0)+'% confidence';
    const si={positive:'😊',neutral:'😐',negative:'😠'};
    const sc={positive:'var(--green)',neutral:'var(--yellow)',negative:'var(--red)'};
    const s=sc[a.sentiment]||'var(--muted)';
    document.getElementById('sentimentIcon').textContent=si[a.sentiment]||'–';
    document.getElementById('sentimentVal').textContent=a.sentiment||'–';
    document.getElementById('sentimentVal').style.color=s;
    document.getElementById('sentimentBar').style.width=(a.sentiment_confidence||0)+'%';
    document.getElementById('sentimentBar').style.background=s;
    document.getElementById('sentimentConf').textContent=(a.sentiment_confidence||0)+'% confidence';
    const rc={high:'var(--red)',medium:'var(--yellow)',low:'var(--green)'};
    const r=rc[a.escalation_risk]||'var(--muted)';
    document.getElementById('cardRisk').className='card'+(a.escalation_risk==='high'?' danger':'');
    document.getElementById('riskDot').style.cssText=`background:${r};animation:${a.escalation_risk==='high'?'pulse 1.5s infinite':'none'}`;
    document.getElementById('riskVal').textContent=(a.escalation_risk||'–').toUpperCase();
    document.getElementById('riskBadge').style.cssText+=`background:${r}22;color:${r};border:1px solid ${r}55;`;
    document.getElementById('signalsList').innerHTML=(a.escalation_signals||[]).map(s=>`<div class="signal-item">"${s}"</div>`).join('');
    document.getElementById('summaryVal').textContent=a.summary||'–';
    document.getElementById('phrasesWrap').innerHTML=(a.key_phrases||[]).map(p=>`<span class="phrase-tag">${p}</span>`).join('');
    document.getElementById('jsonBox').textContent=JSON.stringify({intent:a.intent,topic:a.topic,sentiment:a.sentiment,escalation_risk:a.escalation_risk},null,2);
    const ci=document.getElementById('cardIntent');ci.classList.add('highlight');setTimeout(()=>ci.classList.remove('highlight'),1000);
  }
  function clearChat(){
    chatHistory=[];segments=[];
    document.getElementById('messages').innerHTML='<div class="msg bot"><div class="bubble-wrap"><div class="bubble-label">BOT</div><div class="bubble">👋 Hello! How can I help you today?</div></div><div class="avatar b">B</div></div>';
    document.getElementById('emptyState').style.display='block';
    document.getElementById('analyticsCards').style.display='none';
  }
</script>
</body>
</html>"""


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import webbrowser, threading, time

    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:8001")

    threading.Thread(target=open_browser, daemon=True).start()

    print("\n" + "="*50)
    print("  ConvoBot is starting...")
    print("  Open: http://localhost:8001")
    print("  Press CTRL+C to stop")
    print("="*50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")