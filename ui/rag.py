from ui.globals import *


def load_documents(file_names: list[str], file_paths: list[str]) -> list[Document]:
    """加载多个文档并返回合并后的Document列表"""
    all_docs = []
    for name, path in zip(file_names, file_paths):
        # 根据扩展名选择Loader
        if path.endswith('.pdf'):
            loader = PyPDFLoader(path, mode='single')
        elif path.endswith('.txt'):
            loader = TextLoader(path)
        elif path.endswith('.docx'):
            loader = Docx2txtLoader(path)
        else:
            raise ValueError(f"不支持的文件类型: {path}，目前仅支持 .pdf, .txt 和 .docx")
        # 加载单个文档并合并
        docs = loader.load()
        for doc in docs:
            doc.metadata['file_name'] = name
        all_docs.extend(docs)
        print(f'已加载: {os.path.basename(path)}')
    return all_docs


def split_documents(docs: list[Document]) -> list[Document]:
    for doc in docs:
        doc.page_content = re.sub(r'\s', '', doc.page_content)  # 清除空白符
    splitter = RecursiveCharacterTextSplitter(chunk_size=ses.chunk_size, chunk_overlap=ses.chunk_overlap,
                                              separators=sep_list, length_function=len)
    split_docs = splitter.split_documents(docs)
    return split_docs


def hash_string(string):
    hash_object = hashlib.sha256()
    hash_object.update(string.encode('utf-8'))
    return hash_object.hexdigest()


def connect_to_milvus():
    try:
        connections.connect(alia='default', host='localhost', port=default_server.listen_port)
        print(f'成功连接到 Milvus 服务器，端口：{default_server.listen_port}')
    except Exception as e:
        print(f'连接失败，再次尝试中：{e}')
        default_server.start()
        connections.connect(alia='default', host='localhost', port=default_server.listen_port)
        print(f'成功连接到 Milvus 服务器，端口：{default_server.listen_port}')


def create_contest_clt():
    if 'contest_clt' in utility.list_collections():
        return Collection('contest_clt')
    id = FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=False)
    contest = FieldSchema(name='contest', dtype=DataType.VARCHAR, max_length=256)
    date = FieldSchema(name='date', dtype=DataType.INT64)
    chunk_size = FieldSchema(name='chunk_size', dtype=DataType.INT32)
    chunk_overlap = FieldSchema(name='chunk_overlap', dtype=DataType.INT32)
    dummy = FieldSchema(name='dummy', dtype=DataType.FLOAT_VECTOR, dim=1)
    schema = CollectionSchema(fields=[id, contest, date, chunk_size, chunk_overlap, dummy])
    collection = Collection(name='contest_clt', schema=schema)
    index_params = {
        'metric_type': 'IP',
        'index_type': 'FLAT',
        'params': {}
    }
    collection.create_index(field_name='dummy', index_params=index_params)
    return collection


def create_text_clt():
    if 'text_clt' in utility.list_collections():
        return Collection('text_clt')
    id = FieldSchema(name='id', dtype=DataType.VARCHAR, max_length=128, is_primary=True, auto_id=False)
    contest = FieldSchema(name='contest', dtype=DataType.VARCHAR, max_length=256)
    text = FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=2048)
    embedding = FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=384)
    schema = CollectionSchema(fields=[id, contest, text, embedding])
    collection = Collection(name='text_clt', schema=schema, shard_num=2)
    index_params = {
        'metric_type': 'IP',  # 内积,
        'index_type': 'IVF_FLAT',
        'params': {'nlist': 256}
    }
    collection.create_index(field_name='embedding', index_params=index_params)
    return collection


def retrieval_texts(query, max_ret=5, n_probe=10):
    query_embed = get_text_embeddings(query)
    search_params = {
        'metric_type': 'IP',
        'offset': 0,
        'ignore_growing': False,
        'params': {'nprobe': n_probe}  # 增加 nprobe 值以提高检索范围
    }
    ses.text_clt.load()
    results = ses.text_clt.search(
        data=query_embed,
        anns_field='embedding',
        param=search_params,
        limit=max_ret,
        output_fields=['text'],
        consistency_level='Strong'
    )
    ret_texts = [res.entity.get('text') for res in results[0]]
    return ret_texts


def insert_contest_clt(file_names: list[str]):
    date = int(time())
    chunk_size = ses.chunk_size
    chunk_overlap = ses.chunk_overlap
    id2contest = {}
    for x in file_names:
        x = re.split(r'[_.]', x)
        id2contest[int(x[0])] = x[1]
    new_ids = list(id2contest)
    ses.contest_clt.load()
    results = ses.contest_clt.query(expr=f'id in {new_ids}', output_fields=['id'])
    pre_ids = [res['id'] for res in results]
    add_ids = list(set(new_ids) - set(pre_ids))
    n = len(add_ids)
    print(f'需要插入 {n} 条记录到集合 {ses.contest_clt.name}')
    if n > 0:
        results = ses.contest_clt.insert([add_ids, [id2contest[id] for id in add_ids], [date] * n,
                                         [chunk_size] * n, [chunk_overlap] * n, np.zeros((n, 1))])
        print(f'成功插入 {results.insert_count} 条记录到集合 {ses.contest_clt.name}')
    ses.contest_clt.flush()


def insert_text_clt(file_names: list[str], file_paths: list[str]):
    all_docs = load_documents(file_names, file_paths)
    split_docs = split_documents(all_docs)
    contests = [re.split(r'[_.]', doc.metadata['file_name'])[1] for doc in split_docs]
    texts = [f"《{contest}》：{doc.page_content.strip(sep_str)}" for contest, doc in zip(contests, split_docs)]
    new_ids = [hash_string(text) for text in texts]
    id2info = {new_ids[i]: [contests[i], texts[i]] for i in range(len(new_ids))}
    ses.text_clt.load()
    results = ses.text_clt.query(expr=f'contest in {contests}', output_fields=['id'])
    pre_ids = [res['id'] for res in results]
    add_ids = list(set(new_ids) - set(pre_ids))  # 添加更新后新增的记录
    n = len(add_ids)
    print(f'需要插入 {n} 条记录到集合 {ses.text_clt.name}')
    for i in range(0, n, emb_size):
        j = i + emb_size
        ids = add_ids[i:j]
        contests = [id2info[i][0] for i in ids]
        texts = [id2info[i][1] for i in ids]
        embeddings = get_text_embeddings(texts)
        results = ses.text_clt.insert([ids, contests, texts, embeddings])
        print(f'成功插入 {results.insert_count} 条记录到集合 {ses.text_clt.name}')
    ses.text_clt.flush()


def insert_data(file_names: list[str], file_paths: list[str]):
    insert_contest_clt(file_names)
    insert_text_clt(file_names, file_paths)


def update_contest_clt(file_names: list[str]):
    date = int(time())
    chunk_size = ses.chunk_size
    chunk_overlap = ses.chunk_overlap
    n = len(file_names)
    ids, contests = [], []
    for x in file_names:
        x = re.split(r'[_.]', x)
        ids.append(int(x[0]))
        contests.append(x[1])
    ses.contest_clt.load()
    results = ses.contest_clt.insert([ids, contests, [date] * n,
                                     [chunk_size] * n, [chunk_overlap] * n, np.zeros((n, 1))])
    print(f'成功更新 {results.insert_count} 条记录到集合 {ses.contest_clt.name}')
    ses.contest_clt.flush()


def update_text_clt(file_name: str, file_path: str):
    all_docs = load_documents([file_name], [file_path])
    split_docs = split_documents(all_docs)
    contest = re.split(r'[_.]', file_name)[1]
    texts = [f"《{contest}》：{doc.page_content.strip(sep_str)}" for doc in split_docs]
    new_ids = [hash_string(text) for text in texts]
    id2text = {id: text for id, text in zip(new_ids, texts)}
    new_ids = set(new_ids)
    ses.text_clt.load()
    results = ses.text_clt.query(expr=f'contest == "{contest}"', output_fields=['id'])
    pre_ids = set([res['id'] for res in results])
    del_ids = list(pre_ids - new_ids)  # 删除更新后不存在的记录
    add_ids = list(new_ids - pre_ids)  # 添加更新后新增的记录
    print(f'需要删除的记录数：{len(del_ids)}，新增的记录数：{len(add_ids)}')
    if len(del_ids) > 0:
        results = ses.text_clt.delete(expr=f'id in {del_ids}')
        print(f'成功删除 {results.delete_count} 条记录从集合 {ses.text_clt.name}')
        ses.text_clt.flush()
    n = len(add_ids)
    if n == 0:
        return
    contests = [contest] * n
    texts = [id2text[id] for id in add_ids]
    for i in range(0, n, emb_size):
        j = i + emb_size
        embeddings = get_text_embeddings(texts[i:j])
        results = ses.text_clt.insert([add_ids[i:j], contests[i:j], texts[i:j], embeddings])
        print(f'成功插入 {results.insert_count} 条记录到集合 {ses.text_clt.name}')
    ses.text_clt.flush()


def update_data(file_names: list[str], file_paths: list[str]):
    update_contest_clt(file_names)
    for name, path in zip(file_names, file_paths):
        update_text_clt(name, path)


def delete_contest_clt(contests: list[str]):
    ses.contest_clt.load()
    results = ses.contest_clt.query(expr=f'contest in {contests}', output_fields=['id'])
    ids = [res['id'] for res in results]
    results = ses.contest_clt.delete(expr=f'id in {ids}')
    print(f'成功删除 {results.delete_count} 条记录从集合 {ses.contest_clt.name}')
    ses.contest_clt.flush()


def delete_text_clt(contests: list[str]):
    ses.text_clt.load()
    results = ses.text_clt.query(expr=f'contest in {contests}', output_fields=['id'])
    ids = [res['id'] for res in results]
    results = ses.text_clt.delete(expr=f'id in {ids}')
    print(f'成功删除 {results.delete_count} 条记录从集合 {ses.text_clt.name}')
    ses.text_clt.flush()


def delete_data(contests: list[str]):
    delete_contest_clt(contests)
    delete_text_clt(contests)


def clear_collection():
    collections = utility.list_collections()
    for name in collections:
        utility.drop_collection(name)
        print(f"已删除 Collection: {name}")
    ses.contest_clt = create_contest_clt()
    ses.text_clt = create_text_clt()
    ses.contest_clt.flush()
    ses.text_clt.flush()
