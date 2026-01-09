from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from agent import app  # 질문자님의 에이전트
import json
import asyncio
import uvicorn

api_app = FastAPI()
templates = Jinja2Templates(directory="templates")

@api_app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@api_app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question")

    async def generate():
        inputs = {"question": question, "context": [], "sources": [], "retry_count": 0}
        
        # 노드 단위 스트리밍
        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "generate":
                    answer = value['answer']
                    # 브라우저에 한 번에 쏘지 않고 '전송' 단위를 보냄
                    yield f"data: {json.dumps({'answer': answer})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("main:api_app", host="127.0.0.1", port=8000, reload=True)