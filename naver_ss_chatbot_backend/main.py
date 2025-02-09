from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sqlite3
import os
import sys

from QNA.qna_answer_milvus import answering

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from cfg import cfg


app = FastAPI()

# CORS 설정
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 설정
app.add_middleware(SessionMiddleware, secret_key=cfg.fastapi_secret_key)

DATABASE = cfg.database_name

class Question(BaseModel):
    question: str

def init_db():
    if not os.path.exists(DATABASE):
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('''CREATE TABLE conversations
                     (session_id TEXT, user_question TEXT, bot_response TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app.router.lifespan_context = lifespan

@app.post("/ask")
async def ask(request: Request, question: Question):
    try :
        conversation_history = None
        user_question = question.question
        session_id = request.session.get('session_id')
        print("*"*20)
        print(session_id)
        if not session_id:
            session_id = request.session['session_id'] = os.urandom(24).hex()
            conversation_history = []  # 세션 ID가 없을 때는 새로 초기화
        else:
            # 기존 대화 기록 가져오기
            conn = sqlite3.connect(DATABASE)
            c = conn.cursor()
            c.execute("SELECT user_question, bot_response FROM conversations WHERE session_id = ?", (session_id,))
            conversation_history = c.fetchall()
            conn.close()
        
        # 여기에 OpenAI API를 호출하여 응답을 생성하는 코드를 추가합니다
        # 예시 응답
        print("*"*20)
        print(f"Conversation History : {conversation_history}")
        bot_response = answering(user_question=user_question, conversation_history=conversation_history)

        # 대화 기록 저장
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("INSERT INTO conversations (session_id, user_question, bot_response) VALUES (?, ?, ?)",
                (session_id, user_question, bot_response))
        conn.commit()
        conn.close()
        return JSONResponse(content={"response": bot_response})
    
    except Exception as e:
        return JSONResponse(content={"response": e})
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)