import os
import re
import ast
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
persist_dir = "Cultural_db"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")
web_search_tool = DuckDuckGoSearchResults(num_results=5)

if os.path.exists(persist_dir) and os.listdir(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
else:
    print(f"로컬 DB를 찾을 수 없습니다."); retriever = None

# 2. 그래프 상태(State) 정의
class GraphState(TypedDict):
    question: str
    context: Annotated[List[str], operator.add]
    answer: str
    retry_count: int

# 3. 노드(Nodes) 정의
def retrieve_node(state: GraphState):
    print("\n--- [Node] DB 검색 수행 중 ---")
    if retriever is None:
        return {"context": []}
    
    docs = retriever.invoke(state["question"])
    
    # web_search_node와 형식을 맞춤
    formatted_docs = [
        f"출처 제목: {doc.metadata.get('source', '내부 문서')}\n정보: {doc.page_content}" 
        for doc in docs
    ]
    
    return {
        "context": formatted_docs, 
        "retry_count": 0
        }

def web_search_node(state: GraphState):
    print("\n--- [Node] 웹 검색 수행 중 ---")
    
    # 1. 쿼리 확장 프롬프트 (질문자님의 지침 반영)
    query_gen_prompt = f"""사용자 질문: {state['question']}
당신은 전문 정보 검색원입니다. 위 질문에 대해 정확한 팩트를 찾기 위해 
서로 다른 관점의 최적화된 키워드 검색어 3개를 생성하세요.

지침:
1. 문장 형태가 아닌 키워드 중심으로 구성하세요.
2. 각 검색어는 질문의 핵심 대상과 '종류', '현황', '성분', '역사' 등 서로 다른 속성을 조합하세요.
3. 결과는 반드시 파이썬 리스트 형식으로만 출력하세요. (예: ["키워드1", "키워드2", "키워드3"])

검색어:"""
    
    raw_res = llm.invoke(query_gen_prompt).content.strip()
    try:
        queries = ast.literal_eval(re.search(r"\[.*\]", raw_res, re.DOTALL).group())
    except:
        queries = [state['question']]

    web_contexts = []
    for q in queries:
        print(f"--- [Search Query]: {q} ---")
        try:
            results = web_search_tool.invoke(q)
            # DuckDuckGo 결과 파싱
            items = re.findall(r"snippet: (.*?), title: (.*?), link: (.*?)\]", results)
            for snippet, title, link in items:
                web_contexts.append(f"출처 제목: {title}\n정보: {snippet}")
        except:
            continue

    return {"context": web_contexts}

def generate_node(state: GraphState):
    print("\n--- [Node] 답변 생성 중 ---")
    raw_contexts = state.get("context", [])
    
    # [핵심] 출처 제목이 같은 것들을 하나로 합쳐 중복 제거
    grouped = {}
    for ctx in raw_contexts:
        parts = ctx.split('\n')
        if len(parts) < 2: continue
        title = parts[0].replace("출처 제목: ", "").strip()
        info = parts[1].replace("정보: ", "").strip()
        
        if title not in grouped:
            grouped[title] = []
        if info not in grouped[title]: # 내용 중복도 체크
            grouped[title].append(info)
    
    # 제목별로 내용을 합치고 상위 5개만 번호 매기기
    unique_contexts = []
    for title, infos in grouped.items():
        unique_contexts.append(f"출처 제목: {title}\n정보: {' '.join(infos)}")

    final_contexts = [f"[{i+1}] {ctx}" for i, ctx in enumerate(unique_contexts[:5])]
    context_combined = "\n\n".join(final_contexts)
    
    prompt = [
        ("system", """당신은 전문 분석가입니다. 다음 규칙을 엄격히 준수하여 답변하십시오.

1. 모든 답변은 마크다운(Markdown) 형식을 사용하고, ##와 ###로 구조화하십시오.
2. **출처 인용**: 각 문장의 끝에 해당 정보의 근거가 되는 [데이터]의 번호를 기입하십시오. 
   예시: "규사는 유리의 주성분입니다[1]."
3. **참고 문헌 작성**: 답변의 가장 마지막에 '### 참고 문헌' 섹션을 만드십시오. 
   여기에 사용된 각 번호와 해당 항목의 '출처 제목'을 리스트 형태로 나열하십시오.
4. **엄격한 근거**: 반드시 제공된 [데이터]에 있는 정보만을 바탕으로 답변하십시오. 데이터에 없는 내용을 절대로 지어내지 마십시오.
5. 핵심 용어는 **굵게** 표시하십시오."""),
        
        ("user", f"[데이터]:\n{context_combined}\n\n질문:\n{state['question']}")
    ]
    
    response = llm.invoke(prompt)

    return {
        "answer": response.content.strip(), 
        "retry_count": state.get("retry_count", 0) + 1
    }

# 4. 조건부 엣지(Router) 로직
def grade_documents_router(state: GraphState) -> Literal["generate", "web_search"]:
    print("--- [Edge] 적합성 평가 중 ---")
    
    # [핵심] DB에서 찾은 모든 문서(5개)를 합쳐서 판단
    full_content = "\n".join(state["context"])
    
    score_prompt = f"""질문: {state['question']}\n검색된 데이터 전체:\n{full_content[:2000]}\n
위 데이터들만으로 질문에 대한 충분한 답변이 가능합니까? 
조금이라도 부족하다면 반드시 'NO', 충분하다면 'YES'라고만 답하세요."""
    
    res = llm.invoke(score_prompt)
    decision = res.content.strip().upper()
    
    if "YES" in decision:
        print("--- [Decision] DB 내용 충분: 웹 검색 생략 ---")
        return "generate"
    else:
        print("--- [Decision] DB 내용 부족: 웹 검색 진행 ---")
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