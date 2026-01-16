# --- Imports ---
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from agent import app

# --- App Initialize ---
api_app = FastAPI()

# --- CORS Settings ---
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes ---
@api_app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        question = data.get("question")

        if not question:
            return JSONResponse(status_code=400, content={"answer": "질문을 입력해주세요."})

        # LangGraph 에이전트 호출
        inputs = {
            "question": question, 
            "context": [], 
            "sources": [], 
            "retry_count": 0
        }
        
        result = await app.ainvoke(inputs)
        answer = result.get("answer", "답변을 생성할 수 없습니다.")
        
        return JSONResponse(content={"answer": answer})

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"answer": f"서버 에러가 발생했습니다: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run("main:api_app", host="127.0.0.1", port=8000, reload=True)