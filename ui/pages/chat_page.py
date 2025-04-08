from ui.rag import *

reasoning_content, answer_content = [], []
if 'messages' not in ses:
    ses.messages = []
llm = LLM_names[0]


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
            margin: 100px 0 20px 0;
            animation: floatIn 1s ease-out forwards;
        }
        .big-emoji {
            font-size: 80px;
            margin-bottom: 15px;
            animation: bounce 2s infinite;
        }
        .welcome-text {
            font-size: 20px;
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
    for chunk in response:
        reasoning_chunk = chunk.choices[0].delta.reasoning_content
        answer_chunk = chunk.choices[0].delta.content
        if reasoning_chunk != '':
            reasoning_content.append(reasoning_chunk)
            yield reasoning_chunk
        elif answer_chunk != '':
            answer_content.append(answer_chunk)


def write_reasoning():
    global answer_content
    with st.status('推理过程', expanded=True):
        st.write_stream(reasoning_stream())
    answer_content = ''.join(answer_content)
    st.write(answer_content)


def write_answer():
    global answer_content
    answer_content = st.write_stream(response)


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
    max_ret = st.number_input('最大检索信息数', min_value=1, value=5, key='max_ret')
    n_probe = st.number_input('搜索聚类数', min_value=1, value=10, key='n_probe')

write_messages()

welcome_holder = st.empty()
if len(ses.messages) == 0:
    with welcome_holder:
        welcome()

user_input = st.chat_input('请输入你的问题')

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
    response = llm_client.chat.completions.create(
        model=model,  # ModelScope Model-Id
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_content}
        ],
        stream=True,
    )
    with st.chat_message('ai'):
        if reasoning:
            write_reasoning()
        else:
            write_answer()
    next_msg = {'role': 'ai', 'content': answer_content}
    if reasoning:
        next_msg['reasoning_content'] = ''.join(reasoning_content)
    ses.messages.append(next_msg)
