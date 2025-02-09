############ IMPORT CONFIGS ##############
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from cfg import cfg
#----------------------------------------#

############ IMPORT FUNCTIONS ############
from RAG.milvus_rag import load_pickle_and_save_to_milvus
#----------------------------------------#

######### OpenAI / Milvus Client #########
from openai import OpenAI
from pymilvus import MilvusClient
#----------------------------------------#

import pickle

if __name__ == "__main__":
    pickle_file = cfg.naver_ss_faq
    collection_name = cfg.milvus_collection_name_naver
    openai_client = OpenAI(api_key=cfg.open_ai_key)
    milvus_client = MilvusClient(uri=cfg.milvus_client_uri_naver)
    embedding_dim = cfg.open_ai_embedding_size
    model = cfg.open_ai_model_1

    load_pickle_and_save_to_milvus(
        pickle_file = pickle_file, 
        collection_name = collection_name,
        openai_client = openai_client,
        milvus_client = milvus_client,
        embedding_dim = embedding_dim,
        model = model
        )
