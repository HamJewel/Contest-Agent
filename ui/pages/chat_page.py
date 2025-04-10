from ui.rag import *

ses.reasoning_content, ses.answer_content = [], []


def welcome():
    st.markdown("""
    <style>
        @keyframes floatIn {
            0% {
                transform: translateY(-100px);
                opacity: 0;
            }
            80% {
                transform: translateY(10px);
                opacity: 1;
            }
            100% {
                transform: translateY(0);
            }
        }

        .welcome-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin: 150px 0 20px 0;
            animation: floatIn 1s ease-out forwards;
        }
        .big-emoji {
            font-size: 100px;
            margin-bottom: 5px;
            animation: bounce 2s infinite;
        }
        .welcome-text {
            font-size: 25px;
            font-weight: bold;
        }

        @keyframes bounce {
            0%, 100% {
                transform: translateY(0);
            }
            50% {
                transform: translateY(-10px);
            }
        }
    </style>

    <div class="welcome-container">
        <div class="big-emoji">🤖</div>
        <div class="welcome-text">我是您的智能客服，任何问题都可以咨询我✨</div>
    </div>
    """, unsafe_allow_html=True)


def reasoning_stream():
    for chunk in ses.response:
        reasoning_chunk = chunk.choices[0].delta.reasoning_content
        answer_chunk = chunk.choices[0].delta.content
        if reasoning_chunk != '':
            ses.reasoning_content.append(reasoning_chunk)
            yield reasoning_chunk
        elif answer_chunk != '':
            ses.answer_content.append(answer_chunk)
    ses.reasoning_content = ''.join(ses.reasoning_content)


def write_reasoning():
    status = st.status('推理过程', expanded=True)
    holder = status.empty()
    holder.write_stream(reasoning_stream())
    ses.answer_content = ''.join(ses.answer_content)
    holder.empty()
    status.caption(ses.reasoning_content)
    status.update(state='complete')
    st.write(ses.answer_content)


def write_answer():
    ses.answer_content = st.write_stream(ses.response)


def write_messages():
    for msg in ses.messages:  # 加载缓存
        with st.chat_message(msg['role']):
            if msg['role'] == 'ai':
                if 'reasoning_content' in msg:
                    with st.status('推理过程'):
                        st.caption(msg['reasoning_content'])
                st.write(msg['content'])
            if msg['role'] == 'user':
                st.write(msg['content'])
                with st.status('检索信息'):
                    for text in msg['information']:
                        st.caption(text)


with st.sidebar:
    if st.button('清空聊天记录', type='primary', icon='🗑️', use_container_width=True):
        ses.messages.clear()
    llm = st.selectbox('大模型列表', LLM_names, key='llm')
    model = LLMs[llm]['model']
    reasoning = LLMs[llm]['reasoning']
    max_ret = st.number_input('最大检索信息数', min_value=1, key='max_ret')
    n_probe = st.number_input('搜索聚类簇数', min_value=1, key='n_probe')

write_messages()

welcome_holder = st.empty()
if len(ses.messages) == 0:
    with welcome_holder:
        welcome()

tips = '请输入你的问题' if ses.connected else '请先初始化数据库'
user_input = st.chat_input(tips, disabled=not ses.connected, key='user_input')

if user_input:
    welcome_holder.empty()
    with st.chat_message('user'):
        st.write(user_input)
        with st.status('检索信息', expanded=True):
            ret_texts = retrieval_texts(user_input, max_ret, n_probe)
            for text in ret_texts:
                st.caption(text)
    ses.messages.append({'role': 'user', 'content': user_input, 'information': ret_texts})
    user_content = get_user_content(ret_texts, user_input)
    ses.response = get_chat_completions(model, user_content)
    with st.chat_message('ai'):
        if reasoning:
            write_reasoning()
        else:
            write_answer()
    next_msg = {'role': 'ai', 'content': ses.answer_content}
    if reasoning:
        next_msg['reasoning_content'] = ses.reasoning_content
    ses.messages.append(next_msg)
