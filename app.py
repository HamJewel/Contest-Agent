from ui.globals import *
for k in Cache_keys:
    if k in ses:
        ses[k] = ses[k]
if 'connected' not in ses:
    ses.connected = False
if 'chunk_size' not in ses:
    ses.chunk_size = 300
if 'chunk_overlap' not in ses:
    ses.chunk_overlap = 50
if 'messages' not in ses:
    ses.messages = []
if 'llm' not in ses:
    ses.llm = LLM_names[0]
if 'max_ret' not in ses:
    ses.max_ret = 5
if 'n_probe' not in ses:
    ses.n_probe = 10
st.set_page_config(f'竞赛智能客服机器人', page_icon='assets/robot.png', layout='wide')
st.logo('assets/tipdm.png', size='large')
pages = [
    st.Page(f'ui/pages/data_page.py', title='数据管理', icon='📊'),
    st.Page(f'ui/pages/chat_page.py', title='智能客服', icon='🤖')
]
pg = st.navigation(pages)
pg.run()
