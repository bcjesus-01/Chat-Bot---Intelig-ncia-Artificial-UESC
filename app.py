import os
import streamlit as st
from dotenv import load_dotenv # <-- NOVA IMPORTAÇÃO
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

# Carrega a chave do arquivo oculto .env
load_dotenv() 
DIRETORIO_DB = "./db_computacao"

st.set_page_config(page_title="Assistente CA da Computação", page_icon="💻")
st.title("💻 Assistente Virtual do CACIC - Ciência da Computação UESC")
st.caption("Tire suas dúvidas sobre o curso, ACCs, pré-requisitos e mais!")

@st.cache_resource
def carregar_ia():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DIRETORIO_DB, embedding_function=embeddings)
    
    # O pesquisador que vai no banco de dados
    retriever = db.as_retriever(search_kwargs={"k": 4})
    
    # O Modelo de Linguagem
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    system_prompt = (
        "Você é o assistente virtual oficial do curso de Ciência da Computação da UESC. "
        "Use APENAS os trechos de contexto abaixo para responder à pergunta. "
        "Se a resposta não estiver no contexto, diga que não sabe e oriente o aluno "
        "a procurar o Colegiado ou o Centro Acadêmico. "
        "Não invente informações em hipótese alguma.\n\n"
        "Contexto:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Função auxiliar para juntar os trechos de texto encontrados
    def formatar_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # A MÁGICA DO LCEL: Um pipeline de dados limpo e direto
    rag_chain = (
        {"context": retriever | formatar_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

bot = carregar_ia()

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

pergunta_usuario = st.chat_input("Digite sua dúvida (ex: Como funciona a ACC?)...")

if pergunta_usuario:
    with st.chat_message("user"):
        st.markdown(pergunta_usuario)
    st.session_state.mensagens.append({"role": "user", "content": pergunta_usuario})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Pensando e consultando documentos...")
        
        # Invoca a IA usando a string diretamente
        texto_resposta = bot.invoke(pergunta_usuario)
        
        placeholder.markdown(texto_resposta)
        
    st.session_state.mensagens.append({"role": "assistant", "content": texto_resposta})