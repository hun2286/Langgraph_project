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

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# 1. 환경 설정 및 로드
load_dotenv()
persist_dir = "Cultural_db"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)
web_search_tool = DuckDuckGoSearchResults(num_results=5)

if os.path.exists(persist_dir) and os.listdir(persist_dir):
    vectorstore = Chroma(
        persist_directory=persist_dir, embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
else:
    print(f"로컬 DB를 찾을 수 없습니다.")
    retriever = None


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
        "sources": [doc.metadata.get("source", "알 수 없음") for doc in docs],
        "retry_count": 0,
    }


def web_search_node(state: GraphState):
    print("\n--- [Node] 웹 검색 수행 중 ---")

    # 검색어에 날짜를 넣지 말고 지역명과 날씨 위주로 생성하도록 유도
    query_gen_prompt = f"""사용자 질문: {state['question']}
당신은 전문 정보 검색원입니다. 위 질문에 대해 정확한 팩트를 찾기 위한 최적의 검색어 1개를 생성하세요.
1. 문장 형태가 아닌 키워드 중심으로 구성하세요.
2. 질문의 핵심 대상과 '종류', '현황', '목록'과 같은 명확한 단어를 조합하세요.
3. 검색어는 딱 1개만 출력하세요.

검색어:"""

    search_query = llm.invoke(query_gen_prompt).content.strip().replace('"', "")
    print(f"--- [Search Query]: {search_query} ---")

    results = web_search_tool.invoke(search_query)

    if isinstance(results, str):
        # snippet 부분만 추출하거나 읽기 좋게 줄바꿈 정리
        content_text = results.replace("], [", "]\n\n[").replace(
            "snippet: ", "\n- 정보: "
        )
    elif isinstance(results, list):
        content_text = "\n".join([f"- {res.get('snippet', '')}" for res in results])
    else:
        content_text = str(results)

    print(f"--- [Web Result Success] ---")

    return {
        "context": [f"### [검색된 실시간 정보]\n{content_text}"],
        "sources": ["웹 검색"],
    }


def generate_node(state: GraphState):
    print("\n--- [Node] 답변 생성 중 ---")
    all_contexts = state.get("context", [])
    context_combined = "\n\n".join(all_contexts)

    prompt = [
        (
            "system",
            """당신은 전문 분석가입니다.
1. 모든 답변은 반드시 마크다운(Markdown) 형식을 사용하십시오.
2. 주제에 맞는 적절한 ## 제목과 ### 소제목을 사용하여 구조화하십시오.
3. 핵심 용어는 **굵게** 표시하고, 목록은 불렛 포인트(*)를 사용하십시오.
4. 텍스트를 나열하지 말고, 독자가 한눈에 읽기 편하도록 문단을 나누십시오.
5. 반드시 제공된 [데이터]에 있는 정보만을 바탕으로 답변하십시오. 데이터에 없는 내용을 지어내지 마십시오.""",
        ),
        ("user", f"[데이터]:\n{context_combined}\n\n질문:\n{state['question']}"),
    ]

    response = llm.invoke(prompt)
    return {
        "answer": response.content.strip(),
        "retry_count": state.get("retry_count", 0) + 1,
    }


# 4. 조건부 엣지(Router) 로직
def grade_documents_router(state: GraphState) -> Literal["generate", "web_search"]:
    print("--- [Edge] 적합성 평가 중 ---")

    if not state["context"]:
        return "web_search"

    full_context = "\n\n".join(state["context"])
    score_prompt = f"""질문: {state['question']}
    
[데이터]:{full_context}

당신은 데이터 판독관입니다. 
1. 위 데이터 중 어느 하나라도 질문에 대한 답변 근거를 포함하고 있다면 'YES'라고 하세요.
2. 여러 문서에 정보가 흩어져 있어도 합쳐서 답변이 가능하다면 'YES'입니다.
3. 아예 관련이 없거나 추측해야 하는 경우에만 'NO'라고 하세요.

결과(YES/NO):"""

    res = llm.invoke(score_prompt)
    if "yes" in res.content.strip().lower():
        return "generate"
    return "web_search"


def check_quality_router(state: GraphState) -> Literal["finish", "re_generate"]:
    print("--- [Edge] 품질 검수 중 ---")
    if ("정보 없음" in state["answer"] or len(state["answer"]) < 20) and state[
        "retry_count"
    ] < 2:
        return "re_generate"
    return "finish"


# 5. 그래프 구축
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_conditional_edges(
    "retrieve",
    grade_documents_router,
    {"generate": "generate", "web_search": "web_search"},
)
workflow.add_edge("web_search", "generate")
workflow.add_conditional_edges(
    "generate", check_quality_router, {"finish": END, "re_generate": "generate"}
)

app = workflow.compile()
