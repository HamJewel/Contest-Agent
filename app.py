from ui.globals import *
for k in Cache_keys:
    if k in ses:
        ses[k] = ses[k]
st.set_page_config(f'竞赛智能客服机器人', page_icon='assets/robot.png', layout='wide')
st.logo('assets/tipdm.png', size='large')
pages = [
    st.Page(f'ui/pages/data_page.py', title='数据管理', icon='📊'),
    st.Page(f'ui/pages/chat_page.py', title='智能客服', icon='🤖')
]
pg = st.navigation(pages)
pg.run()
