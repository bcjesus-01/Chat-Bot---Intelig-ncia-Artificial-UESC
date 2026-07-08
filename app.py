import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from operator import itemgetter # <-- Essencial para a nova arquitetura de memória

load_dotenv() 
DIRETORIO_DB = "./db_computacao"

st.set_page_config(page_title="Assistente CA da Computação", page_icon="💻")
st.title("💻 Assistente Virtual do CACIC - Ciência da Computação")
st.caption("Tire suas dúvidas sobre o curso, ACCs, pré-requisitos e mais!")

@st.cache_resource
def carregar_ia():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = Chroma(persist_directory=DIRETORIO_DB, embedding_function=embeddings)
    
    # --- MELHORIA NA BUSCA (MMR) ---
    # fetch_k=20: Pega 20 opções no banco
    # k=6: Seleciona as 6 mais relevantes e com maior diversidade de informação
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # --- PROMPT ATUALIZADO ---
    system_prompt = (
        "Você é o assistente virtual oficial do curso de Ciência da Computação da UESC. "
        "Seu objetivo é dar respostas DETALHADAS e COMPLETAS. "
        "Use APENAS o Contexto Encontrado abaixo e o Histórico da Conversa para responder.\n"
        "Se a resposta não estiver no contexto, diga que não sabe e oriente o aluno a procurar o Colegiado ou o CA. "
        "Nunca invente informações.\n\n"
        "--- Histórico Recente da Conversa ---\n{chat_history}\n\n"
        "--- Contexto Encontrado nos Documentos ---\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    def formatar_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # --- PIPELINE LCEL ATUALIZADO ---
    # Agora ele aceita um dicionário contendo "input" e "chat_history"
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | formatar_docs, 
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history")
        }
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
        
        # --- PREPARANDO A MEMÓRIA ---
        # Pega as últimas 4 mensagens (ignorando a atual) para dar contexto, sem estourar limite
        ultimas_mensagens = st.session_state.mensagens[-5:-1] 
        texto_historico = ""
        for msg in ultimas_mensagens:
            papel = "Aluno" if msg["role"] == "user" else "Assistente"
            texto_historico += f"{papel}: {msg['content']}\n"
            
        if not texto_historico:
            texto_historico = "Nenhuma conversa anterior."
            
        # --- INVOCANDO A IA ---
        texto_resposta = bot.invoke({
            "input": pergunta_usuario,
            "chat_history": texto_historico
        })
        
        placeholder.markdown(texto_resposta)
        
    st.session_state.mensagens.append({"role": "assistant", "content": texto_resposta})