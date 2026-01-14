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
    sources: Annotated[List[str], operator.add]
    answer: str
    retry_count: int

# 3. 노드(Nodes) 정의
def retrieve_node(state: GraphState):
    print("\n--- [Node] DB 검색 수행 중 ---")
    if retriever is None:
        return {"context": [], "sources": [], "retry_count": 0}
    
    docs = retriever.invoke(state["question"])
    
    # web_search_node와 형식을 맞춤
    numbered_docs = [
        f"[{i+1}] 출처 제목: {doc.metadata.get('source', '로컬 DB 문서')}\n정보: {doc.page_content}" 
        for i, doc in enumerate(docs)
    ]
    
    return {
        "context": numbered_docs,
        "sources": [doc.metadata.get('source', '알 수 없음') for doc in docs],
        "retry_count": 0
    }

def web_search_node(state: GraphState):
    print("\n--- [Node] 웹 검색 수행 중 (쿼리 확장 및 출처 정리) ---")
    
    # 1. 쿼리 확장 프롬프트 (질문자님의 지침 반영)
    query_gen_prompt = f"""사용자 질문: {state['question']}
당신은 전문 정보 검색원입니다. 위 질문에 대해 정확한 팩트를 찾기 위해 
서로 다른 관점의 최적화된 키워드 검색어 3개를 생성하세요.

지침:
1. 문장 형태가 아닌 키워드 중심으로 구성하세요.
2. 각 검색어는 질문의 핵심 대상과 '종류', '현황', '성분', '역사' 등 서로 다른 속성을 조합하세요.
3. 결과는 반드시 파이썬 리스트 형식으로만 출력하세요. (예: ["키워드1", "키워드2", "키워드3"])

검색어:"""
    
    # 2. LLM으로부터 검색어 리스트 수신
    raw_query_res = llm.invoke(query_gen_prompt).content.strip()
    
    # 3. 문자열 리스트를 실제 파이썬 리스트로 변환
    try:
        list_str = re.search(r"\[.*\]", raw_query_res, re.DOTALL).group()
        search_queries = ast.literal_eval(list_str)
    except:
        search_queries = [state['question']]

    final_contexts = []
    
    # 4. 3개의 검색어를 각각 순회하며 검색 수행
    for query in search_queries:
        print(f"--- [Search Query]: {query} ---")
        try:
            results = web_search_tool.invoke(query)
            
            if isinstance(results, str):
                # DuckDuckGoSearchResults의 일반적인 출력 형식을 파싱
                items = re.findall(r"\[snippet: (.*?), title: (.*?), link: (.*?)\]", results)
                for snippet, title, link in items:
                    final_contexts.append(f"출처 제목: {title}\n정보: {snippet}")
                    
            elif isinstance(results, list):
                for res in results:
                    title = res.get('title', '알 수 없는 제목')
                    snippet = res.get('snippet', '')
                    final_contexts.append(f"출처 제목: {title}\n정보: {snippet}")
        except Exception as e:
            print(f"검색 중 오류 발생 ({query}): {e}")

    # 5. 중복 제거 및 상위 5개 선택
    unique_contexts = list(dict.fromkeys(final_contexts)) # 내용 중복 제거
    selected_contexts = unique_contexts[:5] # 상위 5개 제한
    
    # 6. 최종적으로 번호 부여
    numbered_contexts = [f"[{i+1}] {ctx}" for i, ctx in enumerate(selected_contexts)]

    # 결과가 없을 경우 방어 로직
    if not numbered_contexts:
        numbered_contexts = ["### [검색 결과]\n관련된 실시간 정보를 찾지 못했습니다."]

    print(f"--- [Web Result Success]: 총 {len(numbered_contexts)}건의 정보 선택 완료 ---")

    return {
        "context": numbered_contexts,
        "sources": ["웹 검색 결과"]
    }

def generate_node(state: GraphState):
    print("\n--- [Node] 답변 생성 중 ---")
    all_contexts = state.get("context", [])
    context_combined = "\n\n".join(all_contexts)
    
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
    
    if not state["context"]:
        return "web_search"
    
    # 지침 강화: "모호하면 NO라고 하라"는 지시 추가
    score_prompt = f"""질문: {state['question']}\n데이터: {state['context'][0][:500]}\n
    위 데이터에 질문에 대한 핵심 정보가 직접적으로 포함되어 있습니까? 
    조금이라도 모호하거나 내용이 부족하면 무조건 'NO'라고 답하세요. (YES/NO)"""
    
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