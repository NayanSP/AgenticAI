import streamlit as st
from lang_backend import chat
from langchain_core.messages import HumanMessage

CONFIG = config = {'configurable':{'thread_id': 'thread 1'}}
if 'msg_history' not in st.session_state:
    st.session_state['msg_history'] = []

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
    
    #ass_ip = chat.invoke({'messages':[HumanMessage(content=user_ip)]}, config=CONFIG)
    #msg = ass_ip['messages'][-1].content
    
    with st.chat_message('assistant'):
        ass = st.write_stream(
            msg_chunk.content for msg_chunk, metadata in chat.stream(
                {'messages': [HumanMessage(content=user_ip)]},
                config=CONFIG,
                stream_mode='messages'
            )
        )
    
    ass_ip = chat.invoke({'messages':[HumanMessage(content=ass)]})