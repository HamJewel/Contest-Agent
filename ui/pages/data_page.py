from ui.rag import *

empty_file_data = pd.DataFrame(columns=['编号', '加载日期', '文件名称'])


def get_file_data():
    ses.file_clt.load()
    results = ses.file_clt.query(expr='date > 0', output_fields=['date', 'file'])
    if len(results) == 0:
        return empty_file_data
    else:
        df = pd.DataFrame(results)
        df['date'] = df['date'].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))
        df.columns = ['加载日期', '文件名称']
        df.insert(0, '编号', range(1, len(df) + 1))
        return df


def init():
    with st.sidebar:
        button = st.button('初始化数据库', type='primary', icon='⚙️', use_container_width=True)
        e1 = st.empty()
        e2 = st.empty()
        if is_connected():
            e1.success('已连接数据库', icon='✅')
        else:
            e1.warning('未连接数据库', icon='⚠️')
        if 'text_clt' in ses:
            e2.success('已获取数据', icon='✅')
        else:
            e2.warning('未获取数据', icon='⚠️')
        if button:
            e1.empty()
            e2.empty()
            e1.info('连接数据库中...', icon='⏳')
            connect_to_milvus()
            e1.empty()
            if not is_connected():
                e1.error('连接失败', icon='❌')
            else:
                e1.success('已连接数据库', icon='✅')
                e2.info('获取数据中...', icon='⏳')
                ses.file_clt = create_file_clt()
                ses.text_clt = create_text_clt()
                ses.file_clt.load()
                ses.text_clt.load()
                ses.file_data = get_file_data()
                e2.empty()
                e2.success('已获取数据', icon='✅')
                return True


init()
col1, col2 = st.columns([2, 1])
col1.write('**已加载文件**')
data_holder = col1.empty()
if 'file_data' in ses:
    data_holder.dataframe(ses.file_data, hide_index=True)
else:
    data_holder.dataframe(empty_file_data, hide_index=True)

fu = col2.file_uploader('📤**上传文件**', ['pdf', 'txt', 'docx'], accept_multiple_files=True)
col3, col4 = col2.columns([1, 1])
clear_data = col3.button('清空数据库', type='primary', icon='🗑️', use_container_width=True)
clear_temp = col4.button('清理临时文件', type='primary', icon='♻️', use_container_width=True)
load = col3.button('加载文件', type='primary', icon='🗃️', disabled=not fu, use_container_width=True)
update = col4.button('更新文件', type='primary', icon='📝', disabled=not fu, use_container_width=True)

if clear_data:
    st.toast('**开始清空数据库**', icon='🚀')
    clear_collection()
    ses.file_data = empty_file_data
    data_holder.dataframe(ses.file_data, hide_index=True)
    st.toast('**数据库清空完成**', icon='🎉')

if clear_temp:
    st.toast('**开始清理临时文件**', icon='🚀')
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path)
    st.toast('**临时文件清理完成**', icon='🎉')

if fu and (load or update):
    names, paths = [], []
    for file in fu:
        names.append(file.name)
        ext = f".{file.name.split('.')[-1]}"
        with NamedTemporaryFile(suffix=ext, dir=temp_path, delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            paths.append(tmp_file.name)
    st.toast('**文件获取完成**', icon='📦')
    if load:
        st.toast('**开始加载数据**', icon='🚀')
        insert_collection(names, paths)
        ses.file_data = get_file_data()
        data_holder.dataframe(ses.file_data, hide_index=True)
        st.toast('**数据加载完成**', icon='🎉')
    if update:
        st.toast('**开始更新数据**', icon='🚀')
        update_collection(names, paths)
        ses.file_data = get_file_data()
        data_holder.dataframe(ses.file_data, hide_index=True)
        st.toast('**数据更新完成**', icon='🎉')
