import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
# from langchain.llms import HuggingFaceHub


def get_pdf_text(pdf_docs):
    raw_text = ''
    for pdf_doc in pdf_docs:
        reader = PdfReader(pdf_doc)
        for page in reader.pages:
            raw_text += page.extract_text()
    return raw_text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store_openai(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store


def get_vector_store_huggingface(chunks):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl")  # , model_kwargs={"device": "opengl"})
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="facebook/m2m100_1.2B", model_kwargs={"temperature": 0.5, "max_length": 512})
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(),
                                                               memory=memory)
    return conversation_chain


def handle_user_question(user_question):
    response = st.session_state.conversation_chain({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i in range(len(st.session_state.chat_history) - 1, -1, -1):
        if i % 2 == 0:
            st.write(user_template.replace('{{MSG}}', st.session_state.chat_history[i].content),
                     unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}', st.session_state.chat_history[i].content),
                     unsafe_allow_html=True)


def main():
    # Load Environment
    load_dotenv()

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # UI
    st.set_page_config(page_title='Chat with multiple PDFs', page_icon=':backpack:')
    st.header('Chat with multiple PDFs :books:')
    user_question = st.text_input('Ask a question about you documents:')

    st.write(css, unsafe_allow_html=True)

    if user_question:
        handle_user_question(user_question)

    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader('Upload your pdf documents and click on process', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner("Processing"):
                # Get PDF(s) Text
                raw_text = get_pdf_text(pdf_docs)

                # Chunk
                chunks = get_text_chunks(raw_text)

                # Vectorize
                vector_store = get_vector_store_openai(chunks)
                # vector_store = get_vector_store_huggingface(chunks)

                # Conversation Chain
                st.session_state.conversation_chain = get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()
