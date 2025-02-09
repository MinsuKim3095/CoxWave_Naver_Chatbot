############ IMPORT CONFIGS ##############
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from cfg import cfg
#----------------------------------------#

####### IMPORT EMBEDDING FUNCTIONS #######
from RAG.milvus_rag import embedding_text
#----------------------------------------#

######### OpenAI / Milvus CLIENT #########
from openai import OpenAI
from pymilvus import MilvusClient
#----------------------------------------#

######### IMPORT PYTHON LIBRARY ##########
import json
#----------------------------------------#


openai_client = OpenAI(api_key=cfg.open_ai_key)
milvus_client = MilvusClient(uri=cfg.milvus_client_uri_naver)
collection_name = cfg.milvus_collection_name_naver
embedding_dim = cfg.open_ai_embedding_size
emb_model = cfg.open_ai_model_1
chat_model = cfg.open_ai_model_2

def answering(user_question, conversation_history):
    search_res = milvus_client.search(
        collection_name = collection_name,
        data=[
            embedding_text(openai_client, user_question, emb_model)
        ],  
        anns_field="question_vector",
        limit=3,  # Return top 3 results
        search_params={"metric_type": "IP"},
        #search_params={"metric_type": "IP", "params": {"anns_field": "question_vector"}},
        output_fields=["question", "answer", "question_vector","answer_vector"],  
    )

    retrieved_lines_with_distances = [
        (res["entity"]["question"], res["entity"]["answer"], res["distance"]) for res in search_res[0]
    ]

    print("*"*20)
    print("Milvus 참고 문서")
    print(json.dumps(retrieved_lines_with_distances, ensure_ascii=False, indent=4))

    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )
    print("*"*20)
    print("Context")
    print(context)

    SYSTEM_PROMPT = \
    """
    1. 당신은 네이버 스마트 스토어 사용자에게 정보를 전달하는 챗봇입니다. 
    2. 주어진 문의의 키워드와 내용을 바탕으로 RAG에 있는 문서에서 답을 찾아서 답변하십시오.
    3. 주어진 문의의 키워드와 내용을 바탕으로 답변 하되, 둘의 관련이 적다면 키워드를 바탕으로 답변하십시오.
    4. 아래 답변 예시를 참고하여, 문의에 대한 답과 함께 사용자에게 후속적으로 이루어져야 하는 
    내용에 대한 안내가 필요한지 질문 하십시오.
    "네이버 스마트스토어는 만 14세 미만의 개인(개인 사업자 포함) 또는 법인사업자는 입점이 불가함을 양해 부탁 드립니다.
    등록에 필요한 서류 안내해드릴까요? 등록 절차는 얼마나 오래 걸리는지 안내가 필요하신가요?"
    - 정확하고 유용한 답변을 제공할 수 있는 다양한 방법을 고려해야 합니다.
    예를 들어:
    - 사용자가 "평균 배송일은 어떻게 되나요?"라고 묻는다면, 이를 "평균 배송일이 얼마나 걸리나요?"로 해석하고 적절한 답변을 제공하세요.
    - 사용자가 "이 제품의 가격은 어떻게 되나요?"라고 묻는다면, 이를 "이 제품의 가격이 얼마인가요?"로 해석하고 적절한 답변을 제공하세요.
    - 사용자가 "배송 시간은 어떻게 되나요?"라고 묻는다면, 이를 "배송 시간이 얼마나 걸리나요?"로 해석하고 적절한 답변을 제공하세요.
    - 사용자가 "오늘보고서는 뭔가요?" 와 같이, 특정 기능에 대해 묻는다면 이를 "오늘보고서의 기능" 으로 해석하고 적절한 답변을 제공하세요.
    - 사용자가 "스마트 스토어의 주문 취소 및 환불 절차는 어떻게 되나요?" 와 같이 "스마트 스토어" 단어를 넣는다면, 이를 "주문 취소 및 환불 절차는 어떻게 되나요?" 으로 해석하고 적절한 답변을 제공하세요.
    5. 항상 사용자의 질문 의도를 이해하고 관련성 있는 정확한 답변을 제공해야 합니다.
    6. 사용자의 질문에 "카테고리"와 "방법" 으로 분류 한 뒤, 내용을 합쳐서 답변합니다.
    예를 들어:
    - 사용자가 "판매 통계 분석 기능은 어떻게 이용하나요?" 라고 묻는다면, 이를 "판매 통계 분석" 카테고리와 "기능 이용" 으로 분류 한뒤,
    이를 "판매 통계 분석 관련 기능 사용법" 에 대한 내용으로 답변합니다.
    - 그럼에도 문맥이 맞지 않을 경우,"저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."
    와 같이 안내 메시지를 출력하십시오.
    """

    conversation_str = "\n".join([f"User: {q}\nBot: {r}" for q, r in conversation_history])
    USER_PROMPT = \
    f"""
    다음 정보를 <context> 태그에 포함하여, <question> 태그에 포함된 질문에 대한 답변을 제공합니다.
    <context>
    {context}
    </context>
    <question>
    {user_question}
    </question>
    """

    response = openai_client.chat.completions.create(
        model=cfg.open_ai_model_2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )

    print("*"*20)
    print("gpt-4o-mini 답변 내역")
    print(response.choices[0].message.content)
    print("*"*20)
    return response.choices[0].message.content
