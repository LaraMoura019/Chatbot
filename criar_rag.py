import os
import glob
import shutil
from pathlib import Path

# lê PDFs página a página
import pdfplumber 
# divide texto em chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter 
# base de dados vetorial  
from langchain_chroma import Chroma      
# converte texto em vetores numéricos              
from langchain_ollama import OllamaEmbeddings  
# estrutura de dados do LangChain        
from langchain_core.documents import Document          


def carregar_pdfs(pasta_pdfs):
    """Lê os manuais médicos em PDF"""
    documentos = []
    ficheiros_pdf = glob.glob(os.path.join(pasta_pdfs, "**", "*.pdf"), recursive=True)

    if not ficheiros_pdf:
        print(f"Nenhum PDF encontrado em: {pasta_pdfs}")
        return []

    for caminho_pdf in ficheiros_pdf:
        nome_ficheiro = Path(caminho_pdf).name      
        categoria = Path(caminho_pdf).parent.name 
        print(f"A carregar: [{categoria}] {nome_ficheiro}")

        try:
            with pdfplumber.open(caminho_pdf) as pdf:
                for num_pagina, pagina in enumerate(pdf.pages, start=1):
                    texto = pagina.extract_text()
                    if texto and texto.strip():         
                        documentos.append(Document(
                            page_content=texto.strip(),
                            metadata={
                                "fonte": nome_ficheiro,
                                "categoria": categoria,
                                "pagina": num_pagina,
                                "tipo": "conhecimento_medico"
                            }
                        ))
        except Exception as e:
            print(f"Erro ao ler {nome_ficheiro}: {e}")
            continue

    print(f"\n{len(documentos)} páginas de PDF carregadas")
    return documentos    

def carregar_transcricao(texto_transcricao, nome_audio, id_paciente, data_consulta, tema):
    """Cria o documento da consulta com os dados do paciente"""
    return [Document(
        page_content=texto_transcricao,
        metadata={
            "fonte": nome_audio,
            "paciente_id": id_paciente,    
            "data": data_consulta,         
            "tema": tema,                  
            "tipo": "consulta_medica",
            "pagina": 1
        }
    )]

def dividir_em_chunks(documentos):
    """Divide os documentos em pedaços menores"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documentos)
    print(f"{len(documentos)} documentos → {len(chunks)} chunks")
    return chunks

# =====================================================================
# FUNÇÕES DE GESTÃO DA BASE DE DADOS (O SEGREDO PARA VÁRIAS CONSULTAS)
# =====================================================================

def inicializar_base_medica(pasta_pdfs, pasta_db="./chroma_db", modelo_embeddings="nomic-embed-text"):
    """
    PASSO 1: Roda APENAS na primeira vez (ou quando adicionar PDFs novos).
    Cria a base de dados com os manuais médicos.
    """
    print(f"A inicializar modelo de embeddings '{modelo_embeddings}'...")
    embeddings = OllamaEmbeddings(model=modelo_embeddings)

    db_existe = os.path.exists(pasta_db) and os.listdir(pasta_db)
    
    if db_existe:
        print(f"A base médica já existe em: {pasta_db}. Apenas carregando...")
        return Chroma(persist_directory=pasta_db, embedding_function=embeddings)
    else:
        print("Base médica vazia. A criar do zero a partir dos PDFs...")
        docs_pdf = carregar_pdfs(pasta_pdfs)
        
        if not docs_pdf:
            print("Nenhum PDF para processar. Crie a pasta e coloque os PDFs.")
            return None
            
        chunks = dividir_em_chunks(docs_pdf)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=pasta_db
        )
        print(f"Base médica criada com sucesso! ({vector_store._collection.count()} chunks indexados)")
        return vector_store

def adicionar_nova_consulta_ao_rag(pasta_db, texto_transcricao, nome_audio, id_paciente, data_consulta, tema, modelo_embeddings="nomic-embed-text"):
    """
    PASSO 2: Roda após cada consulta terminar.
    Injeta a nova transcrição na base que já tem os PDFs.
    """
    embeddings = OllamaEmbeddings(model=modelo_embeddings)
    vector_store = Chroma(persist_directory=pasta_db, embedding_function=embeddings)
    
    docs_consulta = carregar_transcricao(texto_transcricao, nome_audio, id_paciente, data_consulta, tema)
    chunks = dividir_em_chunks(docs_consulta)
    
    vector_store.add_documents(chunks)
    print(f"Nova consulta ({data_consulta}) do paciente {id_paciente} adicionada com sucesso!")
    
    return vector_store

def criar_retriever(vector_store, id_paciente=None, k=5):
    """
    PASSO 3: O motor de busca.
    Agora recebe o id_paciente como variável! Se passado, filtra só para ele.
    """
    filtros = {}
    
    if id_paciente:
        # Se pedirmos um paciente específico, traz o histórico dele OU manuais médicos
        filtros = {
            "$or": [
                {"paciente_id": id_paciente},
                {"tipo": "conhecimento_medico"}
            ]
        }
    
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
            "lambda_mult": 0.7,
            "filter": filtros if filtros else None # Aplica o filtro se existir
        }
    )
    
    return retriever # <-- Faltava isto no seu código!


# =====================================================================
# EXEMPLO DE COMO USAR ISTO NO DIA A DIA
# =====================================================================

#if __name__ == "__main__":
#    # 1. Carrega os manuais (Rápido se já foi feito antes)
#    vs = inicializar_base_medica("./manuais_medicos")
#    
#    if vs:
#        # 2. Faz de conta que tivemos uma consulta hoje
#        texto_da_consulta_de_hoje = "O paciente relata tonturas ao tomar a metformina. Glicemia em jejum a 110."
#        
#        # Guardamos a consulta no banco de dados (fica para sempre!)
#        adicionar_nova_consulta_ao_rag(
#            pasta_db="./chroma_db",
#            texto_transcricao=texto_da_consulta_de_hoje,
#            nome_audio="audio_01_maio.mp3",
#            id_paciente="PAC-001",
#            data_consulta="2026-05-01",
#            tema="diabetes"
#        )
#        
#        # 3. Quando for fazer perguntas ao LLM, cria o retriever com o ID do paciente
#        # Assim o LLM não confunde o PAC-001 com as consultas de outro paciente!
#        retriever_paciente_1 = criar_retriever(vs, id_paciente="PAC-001")
#        
#        print("\nTudo pronto! O sistema já pode cruzar os PDFs com o histórico do PAC-001.")