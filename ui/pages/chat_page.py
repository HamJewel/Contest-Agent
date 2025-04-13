from ui.rag import *

ses.reasoning_content, ses.answer_content = [], []


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
            if msg['role'] == 'assistant':
                if 'reasoning_content' in msg:
                    with st.status('推理过程'):
                        st.caption(msg['reasoning_content'])
                st.write(msg['content'])
            if msg['role'] == 'user':
                st.write(msg['content'])
                if 'info' in msg:
                    with st.status('检索信息'):
                        for text in msg['info']:
                            st.caption(text)
                elif 'file' in msg:
                    with st.status('文件信息'):
                        st.caption(msg['file'])


def query_from_knowledge():
    with st.chat_message('user'):
        st.write(ses.query.text)
        with st.status('检索信息', expanded=True):
            ret_texts = retrieval_texts(ses.query.text, max_ret, n_probe)
            for text in ret_texts:
                st.caption(text)
    msg = {'role': 'user', 'content': ses.query.text}
    ses.dialogs.append(msg)
    msg['info'] = ret_texts
    ses.messages.append(msg)
    user_content = get_user_content(ret_texts, ses.query.text)
    request = [sys_msg, *(ses.dialogs if dialog else []), {'role': 'user', 'content': user_content}]
    ses.response = get_chat_completions(model, request)


def query_from_input():
    with st.chat_message('user'):
        st.write(ses.query.text)
        if ses.query.files:
            file = ses.query.files[0]
            file_info = f'**📄{file.name}**'
            ext = f".{file.name.split('.')[-1]}"
            with NamedTemporaryFile(suffix=ext, dir=temp_path, delete=False) as tmp_file:
                tmp_file.write(file.getvalue())
                doc_content = PyPDFLoader(tmp_file.name, mode='single').load()[0].page_content
            clear_temp_files()
            with st.status('文件信息', expanded=True):
                st.caption(file_info)
    msg = {'role': 'user', 'content': ses.query.text}
    ses.dialogs.append(msg)
    if ses.query.files:
        msg['file'] = file_info
    ses.messages.append(msg)
    user_content = ses.query.text + (f'\n<竞赛文件内容如下>\n{doc_content}' if ses.query.files else '')
    request = [*(ses.dialogs if dialog else []), {'role': 'user', 'content': user_content}]
    ses.response = get_chat_completions(model, request)


with st.sidebar:
    if st.button('清空对话记录', type='primary', icon='🗑️', use_container_width=True):
        ses.messages.clear()
        ses.dialogs = []
    knowledge = st.checkbox('启用知识库', key='knowledge')
    dialog = st.checkbox('启用多轮对话', key='dialog')
    llm = st.selectbox('大模型列表', LLM_names, key='llm',
                       help='1、支持推理的模型能输出详细的推理过程\n2、若回答超时，请更换其他模型(免费API稳定性较差)')
    model = LLMs[llm]['model']
    reasoning = LLMs[llm]['reasoning']
    max_ret = st.number_input('最大检索信息数', min_value=1, key='max_ret')
    n_probe = st.number_input('搜索聚类簇数', min_value=1, key='n_probe')

write_messages()

welcome_holder = st.empty()
if len(ses.messages) == 0:
    with welcome_holder:
        get_welcome_style('🤖', '我是您的智能客服，任何问题都可以咨询我✨')

tips = '请输入你的问题' if ses.connected else '请先初始化数据库'
ses.query = st.chat_input(tips, accept_file=True, file_type=file_type, disabled=not ses.connected)

if ses.query and ses.query.text:
    welcome_holder.empty()
    if not knowledge or ses.query.files:
        query_from_input()
    else:
        query_from_knowledge()
    with st.chat_message('assistant'):
        if reasoning:
            write_reasoning()
        else:
            write_answer()
    msg = {'role': 'assistant', 'content': ses.answer_content}
    ses.dialogs.append(msg)  # 包含用户的问题和模型的最终回答
    if reasoning:
        msg['reasoning_content'] = ses.reasoning_content
    ses.messages.append(msg)  # 包含系统提示词、用户的问题、检索信息(如有)、模型的推理过程(如有)和最终回答
