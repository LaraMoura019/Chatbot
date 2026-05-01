import os
import glob
import shutil
from pathlib import Path
from collections import Counter
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
    """
    Percorre todas as subpastas recursivamente e lê cada PDF página a página.
    Cada página vira um objeto Document com o texto e metadata associada.
    
    Porque página a página e não o PDF inteiro?
    → Porque páginas diferentes falam de coisas diferentes
    → Mais fácil de dividir em chunks depois
    → A metadata guarda de onde veio cada pedaço de texto
    """
    documentos = []
    
    # ** significa "qualquer subpasta a qualquer profundidade"
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

                    # ignora páginas vazias ou só com imagens
                    if texto and texto.strip():         
                        documentos.append(Document(
                            page_content=texto.strip(),
                            
                            # metadata — informação SOBRE o documento
                            # não entra no embedding mas é devolvida com o chunk
                            # útil para o LLM saber de onde vem a informação
                            metadata={
                                "fonte": nome_ficheiro,
                                "categoria": categoria,     # "diabetes" ou "smoking"
                                "pagina": num_pagina,
                                "tipo": "conhecimento_medico"  # distingue de transcrições
                            }
                        ))
        except Exception as e:
            print(f"Erro ao ler {nome_ficheiro}: {e}")
            continue

    # Resumo por categoria para confirmar que tudo foi carregado
    print(f"\n{len(documentos)} páginas carregadas")

    return documentos     

def carregar_transcricao(texto_transcricao, nome_audio):
    """
    Converte a transcrição num Document LangChain.
    
    Porque separado dos PDFs?
    → Para o LLM saber distinguir "isto foi dito pelo médico NA CONSULTA"
      de "isto vem de um manual médico genérico"
    → O metadata tipo="consulta_medica" permite essa distinção
    """
    return [Document(
        page_content=texto_transcricao,
        metadata={
            "fonte": nome_audio,
            "categoria": "consulta",
            "pagina": 1,
            "tipo": "consulta_medica"     # <-- distinção importante
        }
    )]

def dividir_em_chunks(documentos):
    """
    Divide cada Document em pedaços menores (chunks).
    
    Porque dividir?
    → O LLM tem um limite de contexto — não consegue ler 66 páginas de uma vez
    → O retriever é mais preciso com pedaços pequenos e focados
    → "metformina 500mg" está num parágrafo específico, não na página toda
    
    Porque RecursiveCharacterTextSplitter?
    → Tenta dividir por parágrafos primeiro (\n\n)
    → Se ainda for grande, divide por frases (\n)  
    → Se ainda for grande, divide por pontos (.)
    → Último recurso: divide por espaços ( )
    → Nunca corta palavras a meio
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # máximo de 500 caracteres por chunk
                            # (~3-4 frases — informação suficiente sem ser demasiado)
        
        chunk_overlap=50,   # os últimos 50 caracteres de um chunk
                            # repetem-se no início do chunk seguinte
                            # PORQUÊ? Para não perder contexto nas fronteiras
                            # ex: "...tomar metformina | 500mg duas vezes..."
                            #      sem overlap perdias "500mg" do chunk 2
        
        separators=["\n\n", "\n", ".", " "]  # ordem de preferência para dividir
    )

    chunks = splitter.split_documents(documentos)
    
    # Nota: a metadata é automaticamente copiada para cada chunk filho
    # ou seja, cada chunk sabe de que PDF e página veio
    
    print(f"{len(documentos)} documentos → {len(chunks)} chunks")
    return chunks

def criar_vector_store(chunks, pasta_db= "./chroma_db", modelo_embeddings= "nomic-embed-text"):
    """
    Esta é a fase mais importante e mais lenta do RAG.
    
    O que são embeddings?
    → Cada chunk de texto é convertido num vetor de números
    → Ex: "diabetes tipo 2" → [0.23, -0.87, 0.45, 0.12, ...]  (768 números)
    → Textos com significado SEMELHANTE ficam com vetores PRÓXIMOS
    → É isso que permite busca semântica — não por palavras exatas mas por significado
    
    Porque nomic-embed-text?
    → Corre localmente via Ollama (sem enviar dados para a internet)
    → Bom equilíbrio entre qualidade e velocidade
    → Gratuito
    
    Porque ChromaDB?
    → Base de dados especializada em vetores
    → Guarda em disco — não precisas recriar toda a vez
    → Busca os k vetores mais próximos de uma query em milissegundos
    """
    
    print(f"A inicializar modelo de embeddings '{modelo_embeddings}'...")
    embeddings = OllamaEmbeddings(model=modelo_embeddings)
    print(f"Modelo de embeddings carregado!")

    db_existe = os.path.exists(pasta_db) and os.listdir(pasta_db)
    
    if db_existe:
        print(f"A carregar Vector Store existente de: {pasta_db}")
        vector_store = Chroma(
            persist_directory=pasta_db,
            embedding_function=embeddings
        )
        
        total = vector_store._collection.count()
        
        if total == 0:
            print(f"Vector Store vazia! A recriar...")
            shutil.rmtree(pasta_db)    # apaga pasta corrompida
            db_existe = False
        else:
            print(f"Vector Store carregada ({total} chunks indexados)")
            return vector_store

    if not db_existe:
        print(f"A criar embeddings para {len(chunks)} chunks...")
        
        # from_documents faz tudo de uma vez:
        # 1. pega em cada chunk
        # 2. passa pelo modelo de embeddings → vetor numérico
        # 3. guarda (vetor + texto original + metadata) no ChromaDB
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=pasta_db    # guarda em disco automaticamente
        )
        print(f"Vector Store criada ({vector_store._collection.count()} chunks indexados)")

    return vector_store

def criar_retriever(vector_store, k= 5):
    """
    Cria o motor de busca semântica.
    
    Como funciona quando o utilizador faz uma pergunta?
    1. A pergunta é convertida em vetor (com o mesmo modelo de embeddings)
    2. O ChromaDB encontra os k vetores mais próximos
    3. Devolve os chunks correspondentes
    
    Porque MMR (Maximal Marginal Relevance)?
    → Sem MMR: os 5 chunks devolvidos podem ser quase iguais
      (ex: 5 parágrafos diferentes que dizem "metformina trata diabetes")
    → Com MMR: garante DIVERSIDADE — cada chunk traz informação diferente
      (ex: 1 chunk sobre o que é, 1 sobre dosagem, 1 sobre efeitos secundários...)
    
    Parâmetros MMR:
    k=5          → devolve 5 chunks no final
    fetch_k=20   → primeiro busca 20 candidatos por relevância
    lambda_mult  → 1.0 = só relevância, 0.0 = só diversidade, 0.7 = equilíbrio
    """
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

def inicializar_rag(pasta_pdfs, texto_transcricao, nome_audio):
    """
    Lógica corrigida:
    - PDFs → DB persistente (só recria se não existir)
    - Transcrição → adicionada SEMPRE que há uma nova consulta
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # ── 1. PDFs — carrega da DB se já existir ──────────────────
    pasta_db_pdfs = "./chroma_db_pdfs"
    db_pdfs_existe = os.path.exists(pasta_db_pdfs) and os.listdir(pasta_db_pdfs)

    if db_pdfs_existe:
        print("📂 A carregar DB dos PDFs existente...")
        vector_store = Chroma(
            persist_directory=pasta_db_pdfs,
            embedding_function=embeddings
        )
        total = vector_store._collection.count()

        if total == 0:
            print("⚠️  DB vazia! A recriar...")
            shutil.rmtree(pasta_db_pdfs)
            db_pdfs_existe = False
        else:
            print(f"✅ DB dos PDFs carregada ({total} chunks)")

    if not db_pdfs_existe:
        print("🔨 A criar DB dos PDFs...")
        docs_pdf = carregar_pdfs(pasta_pdfs)
        chunks_pdf = dividir_em_chunks(docs_pdf)
        vector_store = Chroma.from_documents(
            documents=chunks_pdf,
            embedding=embeddings,
            persist_directory=pasta_db_pdfs
        )
        print(f"✅ DB dos PDFs criada ({vector_store._collection.count()} chunks)")

    # ── 2. Transcrição — adicionada SEMPRE ────────────────────
    # Cada consulta nova substitui a anterior
    # Para isso apagamos a coleção de transcrições anteriores
    
    print(f"📝 A adicionar transcrição '{nome_audio}'...")
    
    docs_consulta = carregar_transcricao(texto_transcricao, nome_audio)
    chunks_consulta = dividir_em_chunks(docs_consulta)

    # Apaga transcrições anteriores para não acumular consultas antigas
    colecao = vector_store._collection
    ids_consulta = colecao.get(where={"tipo": "consulta_medica"})["ids"]
    
    if ids_consulta:
        print(f"🗑️  A remover {len(ids_consulta)} chunks da consulta anterior...")
        colecao.delete(ids=ids_consulta)

    # Adiciona a transcrição nova
    vector_store.add_documents(chunks_consulta)
    
    total_final = vector_store._collection.count()
    print(f"✅ DB final: {total_final} chunks (PDFs + consulta atual)")

    # ── 3. Teste rápido para confirmar ────────────────────────
    testar_retriever(vector_store, texto_transcricao[:50])

    # ── 4. Retriever ──────────────────────────────────────────
    retriever = criar_retriever(vector_store)
    return retriever, vector_store


# --- Uso ---
#from transcrever import transcricao
#texto = transcricao("./audios/Smoking.mp3", "Smoking.txt")
#retriever, vs = inicializar_rag("./manuais_medicos", texto, "Smoking.mp3")
