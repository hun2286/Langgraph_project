import os
import re
import sys
import operator
import json
from datetime import datetime
from typing import Annotated, List, TypedDict, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, END

# 1. 환경 설정 및 로드
load_dotenv()
persist_dir = "../database/Cultural_db"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")
web_search_tool = DuckDuckGoSearchResults(num_results=3)

if os.path.exists(persist_dir) and os.listdir(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
else:
    print(f"[알림] 로컬 DB를 찾을 수 없습니다."); retriever = None

# 2. 그래프 상태(State) 정의
class GraphState(TypedDict):
    question: str
    context: Annotated[List[str], operator.add]
    sources: Annotated[List[str], operator.add]
    answer: str
    retry_count: int

# 3. 노드(Nodes) 정의
def retrieve_node(state: GraphState):
    print("\n--- [Node] DB 검색 수행 중 ---")
    if retriever is None:
        return {"context": [], "sources": [], "retry_count": 0}
    
    docs = retriever.invoke(state["question"])
    return {
        "context": [doc.page_content for doc in docs],
        "sources": [doc.metadata.get('source', '알 수 없음') for doc in docs],
        "retry_count": 0
    }

def web_search_node(state: GraphState):
    print("\n--- [Node] 웹 검색 수행 중 ---")
    
    # [수정] 검색어에 날짜를 넣지 말고 지역명과 날씨 위주로 생성하도록 유도
    query_gen_prompt = f"""사용자 질문: {state['question']}
위 질문에 대해 가장 최신 정보를 찾을 수 있는 검색어 1개만 생성하세요.
- 날씨의 경우 '지역명 날씨' 형태로 생성하세요 (예: 광주 수기동 날씨)
- 날짜를 검색어에 포함하지 마세요.
검색어:"""
    
    search_query = llm.invoke(query_gen_prompt).content.strip().replace('"', '')
    print(f"--- [Search Query]: {search_query} ---")
    
    results = web_search_tool.invoke(search_query)
    
    # [수정] 데이터가 리스트 형태면 읽기 쉽게 변환
    if isinstance(results, list):
        content_text = "\n".join([f"- {res.get('snippet', '')}" for res in results])
    else:
        content_text = str(results)

    print(f"--- [Raw Result Success] ---")

    return {
        "context": [f"### [검색된 실시간 정보]\n{content_text}"],
        "sources": ["웹 검색"]
    }

def generate_node(state: GraphState):
    print("\n--- [Node] 답변 생성 중 ---")
    all_contexts = state.get("context", [])
    context_combined = "\n\n".join(all_contexts)
    
    prompt = [
        ("system", """당신은 실시간 검색 결과를 바탕으로 답변하는 전문가입니다.
1. [데이터] 섹션의 '검색된 실시간 정보'는 현재 시점의 실제 정보입니다. 
2. 당신의 내부 가이드라인보다 [데이터]의 내용을 우선하십시오.
3. 절대 "기상청을 확인하라"거나 "정보가 없다"는 말을 하지 마십시오.
4. [데이터]에 적힌 날씨, 기온, 하늘 상태 중 하나라도 있다면 그것을 활용해 구체적으로 답하십시오."""), 
        ("user", f"[데이터]:\n{context_combined}\n\n질문:\n{state['question']}")
    ]
    
    response = llm.invoke(prompt)
    return {"answer": response.content.strip(), "retry_count": state.get("retry_count", 0) + 1}

# 4. 조건부 엣지(Router) 로직
def grade_documents_router(state: GraphState) -> Literal["generate", "web_search"]:
    print("--- [Edge] 문서 적합성 평가 중 ---")
    
    if not state["context"]:
        return "web_search"
    
    # DB 내용이 질문과 관련이 있는지 단순 판단
    score_prompt = f"""질문: {state['question']}\n데이터: {state['context'][0][:500]}\n
    위 데이터가 질문에 대답하는 데 직접적인 도움이 되는 내용을 포함하고 있습니까? (YES/NO)"""
    
    res = llm.invoke(score_prompt)
    if "yes" in res.content.strip().lower():
        return "generate"
    return "web_search"

def check_quality_router(state: GraphState) -> Literal["finish", "re_generate"]:
    print("--- [Edge] 품질 검수 중 ---")
    if ("정보 없음" in state["answer"] or len(state["answer"]) < 20) and state["retry_count"] < 2:
        return "re_generate"
    return "finish"

# 5. 그래프 구축
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_conditional_edges("retrieve", grade_documents_router, {"generate": "generate", "web_search": "web_search"})
workflow.add_edge("web_search", "generate")
workflow.add_conditional_edges("generate", check_quality_router, {"finish": END, "re_generate": "generate"})

app = workflow.compile()