import os

########### OPEN AI CONFIG ############
open_ai_key = os.environ["OA_KEY"]
open_ai_model_1 = "text-embedding-3-small"
open_ai_model_2 = "gpt-4o-mini"
open_ai_embedding_size = 1536 # or 512

########### RAW DATA FILE ##########
naver_ss_faq = "/home/minsu/data/final_result.pkl"

########### MILVUS CONFIG #########
milvus_client_uri_naver = "/home/minsu/milvusDB/milvus_naver_ss_faq.db"
milvus_collection_name_naver = "naver_ss_faq_collection"

########### SQLite CONFIG #########
database_name = "chatbot.db"

########### FASTAPI CONFIG #########
fastapi_secret_key = "5f9d85c1f3e630f72127d4e695eb0b811601ab8c4f6f02add519469de5b57b35"