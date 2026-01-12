from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from agent import app 
import uvicorn

api_app = FastAPI()
templates = Jinja2Templates(directory="templates")

@api_app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@api_app.post("/chat")
async def chat(request: Request):
    # 1. 클라이언트로부터 질문 데이터 수신 (비동기)
    data = await request.json()
    question = data.get("question")

    # 2. 에이전트 입력값 설정
    inputs = {"question": question, "context": [], "sources": [], "retry_count": 0}
    
    try:
        # 3. 비동기 방식으로 에이전트 실행
        # 이 과정에서 서버의 이벤트 루프가 차단되지 않고 다른 요청을 처리할 수 있습니다.
        result = await app.ainvoke(inputs)

        # 4. 최종 답변 추출
        answer = result.get("answer", "답변을 생성할 수 없습니다.")
        
        # 5. JSON 형식으로 응답 반환
        return JSONResponse(content={"answer": answer})

    except Exception as e:
        print(f"에러 발생: {e}")
        return JSONResponse(
            status_code=500, 
            content={"answer": "서버 처리 중 오류가 발생했습니다."}
        )

if __name__ == "__main__":
    uvicorn.run("main:api_app", host="127.0.0.1", port=8000, reload=True)