import streamlit as st
import os
import tempfile
from faster_whisper import WhisperModel
import ollama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# ==========================================
# 1. CONFIGURAÇÕES DA PÁGINA E CACHE
# ==========================================
st.set_page_config(page_title="Assistente Clínico EBM", page_icon="🩺", layout="centered")
st.title("🩺 Assistente Clínico Pós-Consulta (RAG)")

@st.cache_resource
def carregar_modelos():
    # Carregar Whisper
    whisper = WhisperModel("large-v3", device="cpu", compute_type="int8")
    # Carregar o RAG (ChromaDB)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="./db_medica", embedding_function=embeddings)
    return whisper, vectorstore

modelo_whisper, vectorstore = carregar_modelos()

# ==========================================
# 2. MEMÓRIA DE SESSÃO
# ==========================================
if "texto_consulta" not in st.session_state:
    st.session_state.texto_consulta = None
if "historico_chat" not in st.session_state:
    st.session_state.historico_chat = []

# ==========================================
# 3. FASE DE UPLOAD (A CONSULTA)
# ==========================================
if st.session_state.texto_consulta is None:
    st.markdown("### Passo 1: Carregar a Gravação da Consulta")
    st.info("O sistema já carregou os PDFs médicos na sua memória permanente.")
    
    ficheiro_audio = st.file_uploader("Arraste o áudio da consulta aqui (MP3, WAV)", type=['mp3', 'wav'])

    if ficheiro_audio is not None:
        with st.spinner("A transcrever a consulta... Isto pode demorar uns minutos."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                temp_audio.write(ficheiro_audio.read())
                caminho_temp = temp_audio.name
            
            segmentos, _ = modelo_whisper.transcribe(caminho_temp, beam_size=5, language="en")
            texto_completo = "".join([s.text + " " for s in segmentos])
            
            st.session_state.texto_consulta = texto_completo
            os.remove(caminho_temp)
            st.rerun()

# ==========================================
# 4. FASE DE CHATBOT (RAG HÍBRIDO)
# ==========================================
else:
    st.success("✅ Consulta e Manuais Médicos processados! O médico virtual está pronto.")
    
    # Se for a primeira mensagem, cria o contexto inicial
    if len(st.session_state.historico_chat) == 0:
        st.session_state.historico_chat.append({
            "role": "assistant", 
            "content": "Olá, sou o seu assistente clínico. Com base na sua consulta e nas normas da DGS, como posso ajudar hoje?"
        })

    # Mostrar histórico
    for msg in st.session_state.historico_chat:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # A Pergunta do Paciente
    pergunta_paciente = st.chat_input("Ex: Quais os cuidados a ter com os meus pés?")

    if pergunta_paciente:
        # 1. Mostrar pergunta
        st.chat_message("user").write(pergunta_paciente)
        
        with st.chat_message("assistant"):
            with st.spinner("A cruzar a consulta com a literatura médica..."):
                
                # ==========================================
                # O MOTOR RAG A FUNCIONAR
                # ==========================================
                # Vai buscar os 3 parágrafos dos PDFs mais parecidos com a pergunta
                docs = vectorstore.similarity_search(pergunta_paciente, k=3)
                conhecimento_cientifico = "\n".join([d.page_content for d in docs])

                # O "Super Prompt" que junta as duas realidades
                instrucoes_hibridas = f"""
                És um Assistente Clínico empático em Portugal. Vais responder à pergunta do paciente cruzando DUAS fontes de informação:
                
                FONTE 1 (O CASO DO PACIENTE - Transcrição):
                {st.session_state.texto_consulta}

                FONTE 2 (A CIÊNCIA - Manuais Médicos da DGS):
                {conhecimento_cientifico}

                REGRAS:
                1. Responde sempre em Português de Portugal (PT-PT).
                2. Sê empático e reconfortante.
                3. Responde com base no CASO DO PACIENTE, mas usa a CIÊNCIA para justificar ou complementar a resposta de forma segura.
                4. Nunca digas "A fonte 1 diz isto". Fala naturalmente.
                """

                # Montar o pacote de mensagens (Instruções + Pergunta)
                mensagens_para_ia = [
                    {"role": "system", "content": instrucoes_hibridas},
                    {"role": "user", "content": pergunta_paciente}
                ]

                # Chamar o Ollama
                resposta = ollama.chat(
                    model='llama3:8b', 
                    messages=mensagens_para_ia,
                    options={'temperature': 0.1}
                )
                
                texto_resposta = resposta['message']['content']
                st.write(texto_resposta)
        
        # Guardar conversa
        st.session_state.historico_chat.append({"role": "user", "content": pergunta_paciente})
        st.session_state.historico_chat.append({"role": "assistant", "content": texto_resposta})