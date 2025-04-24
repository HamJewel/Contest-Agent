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
cache_keys = ['messages', 'dialog', 'knowledge', 'dialogs', 'connected', 'contest_clt',
              'text_clt', 'table', 'chunk_size', 'chunk_overlap', 'llm', 'max_ret', 'n_probe']
LLM_names = ['QwQ-32B(支持推理)', 'Qwen2.5-72B', 'DeepSeek-V3', 'DeepSeek-R1(支持推理)', 'Llama3.3-70B', 'Llama3.1-8B']
LLMs = {
    'QwQ-32B(支持推理)': {'model': 'Qwen/QwQ-32B', 'reasoning': True},
    'Qwen2.5-72B': {'model': 'Qwen/Qwen2.5-72B-Instruct', 'reasoning': False},
    'DeepSeek-V3': {'model': 'deepseek-ai/DeepSeek-V3-0324', 'reasoning': False},
    'DeepSeek-R1(支持推理)': {'name': 'DeepSeek-R1', 'model': 'deepseek-ai/DeepSeek-R1', 'reasoning': True},
    'Llama3.3-70B': {'model': 'LLM-Research/Llama-3.3-70B-Instruct', 'reasoning': False},
    'Llama3.1-8B': {'model': 'LLM-Research/Meta-Llama-3.1-8B-Instruct', 'reasoning': False}
}
# 请分别给出“未来校园”智能应用专项赛的赛项名称、赛道、发布时间、报名时间、组织单位、官网。
prompt = r"""你是一个竞赛智能客服，需要结合用户给出的竞赛相关信息和问题，给出简要且准确的回答。
注意：回答时不需要解释思考过程，直接输出回答即可，同时要求回答排版正确美观。
如果文本中没有相关信息，就直接告知“缺乏相关信息，无法回答。”，不能胡编乱造！
首先用户会提出一个问题，你需要结合接下来的背景信息进行回答。
接下来给出RAG处理后的若干段文本，并且位置越靠前的相关度越高，不同段用换行符\n间隔。
每段文本的结构为“《竞赛名称》：竞赛相关信息”，注意：不同竞赛的信息相互独立，不能混淆利用
因RAG检索的文本段可能有缺失信息，所以可以进行适当推理，获取间接信息。
【示例如下】
用户：
<问题>
第X届XX竞赛的举办时间是什么时候？
<信息>
(多段文本材料)
你：
第X届XX竞赛的举办时间是2025年X月X日到X月X日。
【注：如果是带有互动性质的问题(需要你自己去判断)，请忽略信息材料，直接与用户进行聊天互动，不要回答“缺乏相关信息，无法回答”。】"""
sep_list = ['。', '！', '？', '；', '']  # 长句保底机制''
sep_str = ''.join(sep_list)
sys_msg = {'role': 'system', 'content': prompt}
file_type = ['pdf', 'txt', 'docx']
emb_size = 16  # 最大的嵌入批量数
emb_client = OpenAI(
    api_key='310fb7e2423d29764988b18edb7896f786e6441b',
    base_url="https://aistudio.baidu.com/llm/lmapi/v3",  # aistudio 大模型 api 服务域名
)
llm_client = OpenAI(
    api_key='c57442f8-5cae-46d1-b264-0e8091236dfa',  # ModelScope Token
    base_url='https://api-inference.modelscope.cn/v1/',
)


def clear_temp_files():
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
        os.makedirs(temp_path)


def get_user_content(ret_texts: list[str], query):
    ses.text_clt.load()
    info = '\n'.join([text for text in ret_texts])
    user_content = f'<问题>\n{query}\n<信息>\n{info}'
    return user_content


def get_text_embeddings(texts: list[str]):
    response = emb_client.embeddings.create(
        model='embedding-v1',
        input=texts
    )
    embeddings = [val.embedding for val in response.data]
    return np.array(embeddings)


def get_chat_completions(model, request):
    return llm_client.chat.completions.create(
        model=model,  # ModelScope Model-Id
        messages=request,
        stream=True
    )


def get_welcome_style(emoji, text):
    st.markdown(f"""
    <style>
        @keyframes floatIn {{
            0% {{
                transform: translateY(-100px);
                opacity: 0;
            }}
            80% {{
                transform: translateY(10px);
                opacity: 1;
            }}
            100% {{
                transform: translateY(0);
            }}
        }}

        .welcome-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin: 150px 0 20px 0;
            animation: floatIn 1s ease-out forwards;
        }}
        .big-emoji {{
            font-size: 100px;
            margin-bottom: 5px;
            animation: bounce 2s infinite;
        }}
        .welcome-text {{
            font-size: 25px;
            font-weight: bold;
        }}

        @keyframes bounce {{
            0%, 100% {{
                transform: translateY(0);
            }}
            50% {{
                transform: translateY(-10px);
            }}
        }}
    </style>

    <div class="welcome-container">
        <div class="big-emoji">{emoji}</div>
        <div class="welcome-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)
