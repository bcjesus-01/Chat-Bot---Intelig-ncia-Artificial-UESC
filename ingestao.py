import os
import glob
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


# --- CONFIGURAÇÕES ---
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
load_dotenv() 
DIRETORIO_DB = "./db_computacao"
PASTA_PDFS = "./documentos"


# Lista de URLs (não funciona bem)
LISTA_URLS = []

def iniciar_ingestao():
    all_docs = []

    #VARREDURA DOS PDFs
    print(f"--- 1. Varrendo a pasta '{PASTA_PDFS}' em busca de PDFs ---")
    
    # Cria a pasta caso ela não exista ainda
    if not os.path.exists(PASTA_PDFS):
        os.makedirs(PASTA_PDFS)
        print(f"Pasta '{PASTA_PDFS}' criada. Coloque seus PDFs nela.")
        return

    # Busca todos os arquivos .pdf dentro da pasta
    caminhos_pdfs = glob.glob(os.path.join(PASTA_PDFS, "*.pdf"))

    if not caminhos_pdfs:
        print("Nenhum arquivo PDF encontrado na pasta.")
    else:
        for path in caminhos_pdfs:
            try:
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())
                print(f"[OK] Carregado: {os.path.basename(path)}")
            except Exception as e:
                print(f"[ERRO] Falha ao ler {path}: {e}")

    #CARREGAMENTO DE SITES (inutil por enquanto)
    print("\n--- 2. Carregando URLs ---")
    for url in LISTA_URLS:
        try:
            loader = WebBaseLoader(url)
            all_docs.extend(loader.load())
            print(f"[OK] Site carregado: {url}")
        except Exception as e:
            print(f"[ERRO] Falha ao carregar site {url}: {e}")

    if not all_docs:
        print("Sem dados para processar.")
        return

    #FRAGMENTAÇÃO
    print("\n--- 3. Fragmentando Texto ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_docs)

    #CRIAÇÃO DO BANCO VETORIAL
    print(f"\n--- 4. Gerando Banco de Dados Vetorial ({len(chunks)} fragmentos) ---")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DIRETORIO_DB
    )
    
    print("\n--- PROCESSO CONCLUÍDO! ---")
    print(f"O conhecimento de {len(caminhos_pdfs)} PDFs e {len(LISTA_URLS)} sites foi salvo.")

if __name__ == "__main__":
    iniciar_ingestao()