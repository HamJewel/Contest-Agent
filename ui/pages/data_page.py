from ui.rag import *

empty_table = pd.DataFrame(columns=['编号', '添加日期', '竞赛名称', '文本段长度', '段重叠长度'])


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
        <div class="big-emoji">🔗</div>
        <div class="welcome-text">请先初始化数据库🗃️</div>
    </div>
    """, unsafe_allow_html=True)


def update_table():
    global data_holder
    ses.contest_clt.load()
    results = ses.contest_clt.query(expr='id >= 0',
                                    output_fields=['id', 'date', 'contest', 'chunk_size', 'chunk_overlap'])
    if len(results) == 0:
        ses.table = empty_table
    else:
        df = pd.DataFrame(results, columns=['id', 'date', 'contest', 'chunk_size', 'chunk_overlap'])
        df['date'] = df['date'].apply(lambda x: datetime.fromtimestamp(x, tz=zone).strftime("%Y-%m-%d %H:%M:%S"))
        df.columns = ['编号', '添加日期', '竞赛名称', '文本段长度', '段重叠长度']
        ses.table = df
    data_holder.dataframe(ses.table, hide_index=True)


def init_state():
    global init, e1, e2
    flag = 0
    if ses.connected:
        e1.success('已连接数据库', icon='✅')
        flag += 1
    else:
        e1.warning('未连接数据库', icon='⚠️')
    if 'contest_clt' in ses and 'text_clt' in ses:
        e2.success('已获取数据', icon='✅')
        flag += 1
    else:
        e2.warning('未获取数据', icon='⚠️')

    if init:
        e1.empty()
        e2.empty()
        e1.info('连接数据库中...', icon='⏳')
        try:
            connect_to_milvus()
        except Exception as _:
            e1.empty()
            e1.error('连接失败，请重试', icon='❌')
            return False
        ses.connected = True
        e1.empty()
        e1.success('已连接数据库', icon='✅')
        e2.info('获取数据中...', icon='⏳')
        ses.contest_clt = create_file_clt()
        ses.text_clt = create_text_clt()
        e2.empty()
        e2.success('已获取数据', icon='✅')
        return True
    return flag == 2


with st.sidebar:
    if st.button('清理临时文件', type='primary', icon='♻️', use_container_width=True):
        st.toast('**开始清理临时文件**', icon='🚀')
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.makedirs(temp_path)
        st.toast('**临时文件清理完成**', icon='🎉')
    init = st.button('初始化数据库', type='primary', icon='🔗', use_container_width=True)
    status = st.status('初始化状态', expanded=True, state='running')
    e1 = status.empty()
    e2 = status.empty()

if not init_state():
    welcome()
    status.update(expanded=True, state='error')
    st.stop()
status.update(expanded=False, state='complete')

with st.sidebar:
    st.number_input('文本段长度', min_value=10, key='chunk_size')
    st.number_input('段重叠长度', min_value=0, key='chunk_overlap')
col1, col2 = st.columns([2, 1])
col1.write('**已添加文件**')
data_holder = col1.empty()
update_table()

fu = col2.file_uploader('📤**上传文件**', ['pdf', 'txt', 'docx'], accept_multiple_files=True)
col3, col4 = col2.columns([1, 1])
insert = col3.button('添加文件', type='primary', icon='🗃️', disabled=not fu, use_container_width=True)
update = col4.button('更新文件', type='primary', icon='📝', disabled=not fu, use_container_width=True)
col2.divider()
del_names = col2.multiselect('**选择要删除的文件**', ses.table['文件名称'], disabled=ses.table.empty)
delete = col2.button('删除文件', type='primary', icon='🗑️', disabled=len(del_names) == 0, use_container_width=True)
clear = col2.button('清空数据库', type='primary', icon='🧹', disabled=ses.table.empty, use_container_width=True)

if clear:
    st.toast('**开始清空数据库**', icon='🚀')
    clear_collection()
    ses.table = empty_table
    st.toast('**数据库清空完成**', icon='🎉')
    st.rerun()

if delete:
    st.toast('**开始删除文件**', icon='🚀')
    delete_data(del_names)
    st.toast('**文件删除完成**', icon='🎉')
    st.rerun()

if fu and (insert or update):
    names, paths = [], []
    for file in fu:
        names.append(file.name)
        ext = f".{file.name.split('.')[-1]}"
        with NamedTemporaryFile(suffix=ext, dir=temp_path, delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            paths.append(tmp_file.name)
    st.toast('**文件获取完成**', icon='📦')
    if insert:
        st.toast('**开始添加文件**', icon='🚀')
        insert_data(names, paths)
        st.toast('**文件添加完成**', icon='🎉')
    if update:
        st.toast('**开始更新文件**', icon='🚀')
        update_data(names, paths)
        st.toast('**文件更新完成**', icon='🎉')
    st.rerun()
