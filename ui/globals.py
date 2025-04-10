from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import utility, connections, Collection
from tempfile import NamedTemporaryFile
from milvus import default_server
from openai import OpenAI
from streamlit import session_state as ses
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st
from time import time
import pandas as pd
import numpy as np
import hashlib
import shutil
import os
import re
temp_path = 'temp'
os.makedirs(temp_path, exist_ok=True)
zone = ZoneInfo('Asia/Shanghai')
Cache_keys = ['messages', 'connected', 'file_clt', 'text_clt', 'table',
              'chunk_size', 'chunk_overlap', 'llm', 'max_ret', 'n_probe']
LLM_names = ['DeepSeek-V3', 'DeepSeek-R1(支持推理)', 'QwQ-32B(支持推理)', 'Qwen2.5-72B', 'Llama3.3-70B', 'Llama3.1-8B']
LLMs = {
    'DeepSeek-V3': {'model': 'deepseek-ai/DeepSeek-V3-0324', 'reasoning': False},
    'DeepSeek-R1(支持推理)': {'name': 'DeepSeek-R1', 'model': 'deepseek-ai/DeepSeek-R1', 'reasoning': True},
    'QwQ-32B(支持推理)': {'model': 'Qwen/QwQ-32B', 'reasoning': True},
    'Qwen2.5-72B': {'model': 'Qwen/Qwen2.5-72B-Instruct', 'reasoning': False},
    'Llama3.3-70B': {'model': 'LLM-Research/Llama-3.3-70B-Instruct', 'reasoning': False},
    'Llama3.1-8B': {'model': 'LLM-Research/Meta-Llama-3.1-8B-Instruct', 'reasoning': False}
}
prompt = r"""你是一个竞赛智能客服，需要结合用户给出的竞赛相关信息和问题，给出简洁且精确的回答。
首先这些文本里包含了若干段RAG技术筛选后的背景信息，并且位置越靠前的相关度越高，不同段用换行符\n间隔。
每段信息的结构为“《编号_竞赛名称》：竞赛相关信息”，注意：不同竞赛的信息相互独立，不能混淆利用
接下来用户会提出一个问题，你需要结合这些背景信息给出回答。
注意：回答时不需要解释推理过程，只需要直接输出回答即可，同时要求回答排版清晰美观。
如果文本中没有强相关信息，就直接告知“缺乏相关信息，无法回答。”，不能胡编乱造！
【示例如下】
用户：
<信息>
(多段文本材料)
<问题>
第X届XX竞赛的举办时间是什么时候？
你：
第X届XX竞赛的举办时间是2025年X月X日到X月X日。
【注：如果是带有互动性质的问题(需要你自己去判断)，请忽略信息材料，直接与用户进行聊天互动，不要回答“缺乏相关信息，无法回答”。】"""

emb_size = 16  # 最大的嵌入批量数
emb_client = OpenAI(
    api_key='310fb7e2423d29764988b18edb7896f786e6441b',
    base_url="https://aistudio.baidu.com/llm/lmapi/v3",  # aistudio 大模型 api 服务域名
)
llm_client = OpenAI(
    api_key='c57442f8-5cae-46d1-b264-0e8091236dfa',  # ModelScope Token
    base_url='https://api-inference.modelscope.cn/v1/',
)


def get_user_content(ret_texts: list[str], query):
    ses.text_clt.load()
    info = '\n'.join([text for text in ret_texts])
    user_content = f'<信息>\n{info}\n<问题>\n{query}'
    return user_content


def get_text_embeddings(texts: list[str]):
    response = emb_client.embeddings.create(
        model='embedding-v1',
        input=texts
    )
    embeddings = [val.embedding for val in response.data]
    return np.array(embeddings)


def get_chat_completions(model, user_content):
    return llm_client.chat.completions.create(
        model=model,  # ModelScope Model-Id
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_content}
        ],
        stream=True
    )
