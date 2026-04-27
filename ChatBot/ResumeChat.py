import streamlit as st
from lang_backend import chat
from langchain_core.messages import HumanMessage
import uuid

# utility functions
def generate_threadd_id():
    return uuid.uuid4()

def reset_chat():
    thrd_id = generate_threadd_id()
    st.session_state['thread_id'] = thrd_id
    add_thread(st.session_state['thread_id'])
    st.session_state['msg_history'] = []

def load_converse(th_id):
    return chat.get_state(config={'configurable':{'thread_id':th_id}}).values['messages']

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_thread']:
        st.session_state['chat_thread'].append(thread_id)

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_threadd_id()

if 'chat_thread' not in st.session_state:
    st.session_state['chat_thread'] = []

add_thread(st.session_state['thread_id'])

CONFIG = config = {'configurable':{'thread_id':st.session_state['thread_id']}}
if 'msg_history' not in st.session_state:
    st.session_state['msg_history'] = []

#SIdebar UI
st.sidebar.title('LangGraph CHatbot')
if st.sidebar.button('New Chat'):
    reset_chat()
st.sidebar.header('My COnversation')

for th_id in st.session_state['chat_thread']:
    if st.sidebar.button(str(th_id)):
        st.session_state['thread_id'] = th_id
        msg = load_converse(th_id=th_id)

        temp_msg_dict = []

        for m in msg   :
            if isinstance(m, HumanMessage):
                role='user'
            else:
                role = 'assistant'
            temp_msg_dict.append({'role':role,'content':m.content})
        
        st.session_state['message_history'] = temp_msg_dict


# {'role':'user','content':'Hi'}
# {'role':'assistant','content':'Hi, How may I help you?'}

for msg in st.session_state['msg_history']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])


user_ip = st.chat_input("Type Here")

if user_ip:
    st.session_state['msg_history'].append({'role':'user','content':user_ip})
    with st.chat_message('user'):
        st.text(user_ip)
    
    ass_ip = chat.invoke({'messages':[HumanMessage(content=user_ip)]}, config=CONFIG)
    msg = ass_ip['messages'][-1].content
    st.session_state['msg_history'].append({'role':'assistant','content':msg})
    with st.chat_message('assistant'):
        st.text(msg)