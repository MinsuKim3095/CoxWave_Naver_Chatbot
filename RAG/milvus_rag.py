########## Import Milvus Library ##########
from pymilvus import DataType
#------------------------------------------#
from tqdm import tqdm

import re
import pickle

def create_new_milvus_collection(milvus_client, collection_name, embedding_dim):
    """Create New Milvus Collection Using Milvus Client"""
    if milvus_client.has_collection(collection_name):
        print(f"생성하고자 하는 Collection명({collection_name})과 동일한 Collection을 삭제합니다")
        milvus_client.drop_collection(collection_name)

    schema = milvus_client.create_schema(
        auto_id=False,
        enable_dynamic_fields=True
    )
    schema.add_field(field_name="faq_id", datatype=DataType.INT64, is_primary=True, description="faq_id")
    schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=2000, description="question")
    schema.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=2000, description="answer")
    schema.add_field(field_name="question_vector", datatype=DataType.FLOAT_VECTOR, dim=embedding_dim, description="question")
    schema.add_field(field_name="answer_vector", datatype=DataType.FLOAT_VECTOR, dim=embedding_dim, description="answer")

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="question_vector",
        index_type="AUTOINDEX",
        metric_type="IP"
    )
    index_params.add_index(
        field_name="answer_vector",
        index_type="AUTOINDEX",
        metric_type="IP"
    )
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",  # Strong consistency level
        schema=schema,
        index_params=index_params,
        )
    
def clean_text(text):
    if not isinstance(text, str):
        raise TypeError(f"Expected string or bytes-like object, got {type(text).__name__}")
    
    # 불필요한 텍스트 제거
    text = re.sub(r'\n+', ' ', text)  # 줄바꿈 제거
    text = re.sub(r'\xa0+', ' ', text)  # 불필요한 공백 제거
    text = re.sub(r'별점\d점', '', text)  # 별점 제거
    text = re.sub(r'위 도움말이 도움이 되었나요\?', '', text)  # 불필요한 질문 제거
    text = re.sub(r'소중한 의견을 남겨주시면 보완하도록 노력하겠습니다\.', '', text)  # 불필요한 문장 제거
    text = re.sub(r'보내기', '', text)  # 불필요한 텍스트 제거
    text = re.sub(r'도움말 닫기', '', text)  # 불필요한 텍스트 제거
    return text.strip()

def clean_data(data):
    cleaned_data = []
    for key, value in data.items():
        if isinstance(value, dict):
            cleaned_data.extend(clean_data(value))
        elif isinstance(value, str):
            cleaned_data.append({"question": clean_text(key), "answer": clean_text(value)})
    return cleaned_data

def embedding_text(openai_client, text, model):
    """Create Embedding Using OpenAI"""
    emb_result = openai_client.embeddings.create(input=text, model=model).data[0].embedding
    return emb_result

def split_text(text, max_tokens=8192):
    """텍스트를 최대 토큰 길이에 맞게 분할"""
    tokens = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for token in tokens:
        token_length = len(token)
        if current_length + token_length + 1 > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [token]
            current_length = token_length + 1
        else:
            current_chunk.append(token)
            current_length += token_length + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def insert_data_to_milvus(openai_client, data, model, milvus_client, collection_name):
    """Insert Data to Milvus Collection"""
    _data = []
    for i, item in enumerate(tqdm(data, desc=f"Creating embeddings to {collection_name}")):
        # print(item['question'])
        # print("*"*20)
        # print(item['answer'])
        question_chunks = split_text(item['question'])
        answer_chunks = split_text(item['answer'])
        for qc in question_chunks:
            question_vector = embedding_text(openai_client, qc, model)
            for ac in answer_chunks:
                answer_vector = embedding_text(openai_client, ac, model)
                _data.append({
                    "faq_id": i,
                    "question": qc,
                    "answer": ac,
                    "question_vector": question_vector,
                    "answer_vector": answer_vector
                })

    milvus_client.insert(collection_name=collection_name, data=_data)

def load_pickle_and_save_to_milvus(pickle_file,collection_name, openai_client, milvus_client, embedding_dim, model):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    
    cleaned_data = clean_data(data)
    
    create_new_milvus_collection(milvus_client, collection_name, embedding_dim=embedding_dim)  # Assuming embedding dimension is 768
    insert_data_to_milvus(openai_client, cleaned_data, model, milvus_client, collection_name)
