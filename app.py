import os
import hashlib
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agente import criar_agente
from criar_rag import inicializar_base_medica, adicionar_nova_consulta_ao_rag, criar_retriever
from transcrever import transcricao
from voz import transcrever_pergunta, sintetizar_resposta, obter_modelo_whisper

st.set_page_config(
    page_title="Clara – Assistente de Saúde",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #f4f1ec 0%, #eef3f8 60%, #f0ece4 100%) !important;
    font-family: 'Nunito', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
h1 {
    font-family: 'Lora', serif !important;
    font-size: 2.4rem !important;
    color: #2d4a6b !important;
    letter-spacing: -0.3px;
    margin-bottom: 0 !important;
}
.subtitle {
    font-family: 'Nunito', sans-serif;
    font-size: 1.05rem;
    color: #6b7f8f;
    margin-top: 4px;
    margin-bottom: 14px;
}
hr {
    border: none !important;
    border-top: 2px solid #d8e4ee !important;
    margin: 8px 0 20px 0 !important;
}
[data-testid="stChatMessageContent"] p {
    font-family: 'Nunito', sans-serif !important;
    font-size: 1.08rem !important;
    line-height: 1.8 !important;
    color: #1e2d3d !important;
}
[data-testid="stChatMessage"][data-author="user"] [data-testid="stChatMessageContent"] {
    background: #ddeaf5 !important;
    border-radius: 20px 20px 6px 20px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(45,74,107,0.10);
    border: 1.5px solid #bdd1e6;
}
[data-testid="stChatMessage"][data-author="assistant"] [data-testid="stChatMessageContent"] {
    background: #fdf7ee !important;
    border-radius: 20px 20px 20px 6px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(180,150,100,0.10);
    border: 1.5px solid #e8d9bf;
}
[data-testid="stBottomBlockContainer"],
[data-testid="stBottomBlockContainer"] > div,
[data-testid="stBottomBlockContainer"] > div > div,
[data-testid="stChatInput"],
section[data-testid="stChatInput"],
.stChatFloatingInputContainer,
.stChatFloatingInputContainer > div,
.stChatInputContainer {
    background: #f4f1ec !important;
    background-color: #E8F0F7 !important;
    box-shadow: none !important;
    border-top: none !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Nunito', sans-serif !important;
    font-size: 1.05rem !important;
    color: #1e2d3d !important;
    background: #ffffff !important;
    border-radius: 20px !important;
    border: 2px solid #c2d5e8 !important;
    padding: 14px 22px !important;
    box-shadow: 0 2px 10px rgba(45,74,107,0.08) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #7aa5c8 !important;
    box-shadow: 0 0 0 3px rgba(122,165,200,0.18) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #9aafbf !important;
    font-size: 1rem !important;
}
[data-testid="stSidebar"] {
    background: #e8f0f7 !important;
    border-right: 2px solid #ccdde8 !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Lora', serif !important;
    color: #2d4a6b !important;
    font-size: 1.25rem !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: #3a4f5e !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: #e8f4fd !important;
    border: 1.5px solid #b5d3ea !important;
    border-radius: 12px !important;
    color: #2d4a6b !important;
    font-size: 0.97rem !important;
}
[data-testid="stSidebar"] input {
    border-radius: 10px !important;
    border: 2px solid #bdd1e6 !important;
    background: #f8fbff !important;
    color: #1e2d3d !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 1rem !important;
    padding: 8px 12px !important;
}
[data-testid="stFileUploader"] {
    border: 2.5px dashed #7aa5c8 !important;
    border-radius: 14px !important;
    background: #f0f7fd !important;
    padding: 10px !important;
}
[data-testid="stSidebar"] button {
    background: #a8c8e8 !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 16px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 900 !important;
    font-size: 1.18rem !important;
    padding: 14px 20px !important;
    letter-spacing: 0.2px;
    transition: all 0.2s ease;
    box-shadow: 0 3px 10px rgba(100,150,200,0.20) !important;
    width: 100% !important;
}
[data-testid="stSidebar"] button:hover {
    background: #8fb8de !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 16px rgba(100,150,200,0.30) !important;
}
.voz-label {
    font-family: 'Nunito', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #2d4a6b;
    margin-bottom: 6px;
    display: block;
}
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# CABEÇALHO
# ==========================================
st.markdown("# 🩺 Clara")
st.markdown(
    '<p class="subtitle">Que dúvidas quer esclarecer hoje?</p>',
    unsafe_allow_html=True
)
st.markdown("---")


# ==================
# GESTÃO DE MEMÓRIA
# ==================
if "todas_as_consultas" not in st.session_state:
    st.session_state.todas_as_consultas = {}

if "consulta_atual" not in st.session_state:
    st.session_state.consulta_atual = None

if "historico_ia" not in st.session_state:
    st.session_state.historico_ia = []

if "mensagens_ecra" not in st.session_state:
    st.session_state.mensagens_ecra = [
        {
            "role": "assistant",
            "content": (
                "Olá! 👋 Sou a **Clara**, a sua assistente de saúde.\n"
                "Estou aqui para o ajudar a esclarecer as dúvidas sobre a sua consulta.\n"
                "Pode escrever a sua pergunta ou usar o microfone 🎤 aqui em baixo."
            )
        }
    ]

if "executor" not in st.session_state:
    st.session_state.executor = None

if "resposta_audio" not in st.session_state:
    st.session_state.resposta_audio = None

if "ultimo_audio_hash" not in st.session_state:
    st.session_state.ultimo_audio_hash = None


# ================
# BARRA LATERAL
# ================
with st.sidebar:
    st.markdown("## Consultas Anteriores")

    if not st.session_state.todas_as_consultas:
        st.info("Ainda não tem consultas guardadas.")
    else:
        for nome_sessao in st.session_state.todas_as_consultas.keys():
            if st.button(f"💬 {nome_sessao}", use_container_width=True):
                st.session_state.consulta_atual = nome_sessao
                st.session_state.executor      = st.session_state.todas_as_consultas[nome_sessao]["executor"]
                st.session_state.historico_ia  = st.session_state.todas_as_consultas[nome_sessao]["historico_ia"]
                st.session_state.mensagens_ecra = st.session_state.todas_as_consultas[nome_sessao]["mensagens_ecra"]
                st.session_state.resposta_audio = None
                st.rerun()

    st.markdown("---")
    st.markdown("## Carregar Consulta")
    st.info(
        "Esta área é para um familiar ou cuidador.\n"
        "Siga os passos abaixo para carregar a consulta no sistema."
    )

    st.markdown("---")
    st.markdown("**🎙️ Passo 1 — Ficheiro de Áudio**")
    ficheiro_audio = st.file_uploader(
        "Arraste ou selecione o áudio da consulta",
        type=["mp3", "wav", "m4a"],
        label_visibility="collapsed"
    )

    st.markdown("**👤 Passo 2 — Dados do Paciente**")
    id_paciente   = st.text_input("Nome ou número do paciente", value="Paciente A", placeholder="Ex: Manuel Costa")
    tema_consulta = st.text_input("Tema da consulta",           value="Geral",      placeholder="Ex: Cardiologia")
    data_consulta = st.date_input("Data da consulta")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Carregar Consulta no Sistema"):
        if ficheiro_audio is not None:
            with st.spinner("A processar a consulta… Um momento, por favor."):
                os.makedirs("./audios", exist_ok=True)

                caminho_audio_temp = f"./audios/{ficheiro_audio.name}"
                with open(caminho_audio_temp, "wb") as f:
                    f.write(ficheiro_audio.getbuffer())

                vs = inicializar_base_medica("./manuais_medicos")
                nome_txt = ficheiro_audio.name.replace(".", "_") + ".txt"
                
                # --- O QUE TEM DE MUDAR É ISTO AQUI: ---
                modelo_memoria = obter_modelo_whisper() # 1. Vai buscar o modelo à memória
                texto_transcrito = transcricao(
                    ficheiro_audio=caminho_audio_temp, 
                    ficheiro_txt=nome_txt,
                    model=modelo_memoria,               # 2. Passa o modelo para a função
                    language="pt"
                )
                # --------------------------------------
                vs_atualizado = adicionar_nova_consulta_ao_rag(
                    pasta_db="./chroma_db",
                    texto_transcricao=texto_transcrito,
                    nome_audio=ficheiro_audio.name,
                    id_paciente=id_paciente,
                    data_consulta=str(data_consulta),
                    tema=tema_consulta
                )

                retriever_do_paciente = criar_retriever(vs_atualizado, id_paciente=id_paciente, tema_consulta=tema_consulta)
                executor = criar_agente(retriever_do_paciente, vs_atualizado, id_paciente, tema_consulta)

                nome_da_sessao = f"{tema_consulta} - {data_consulta}"
                mensagens_iniciais = [
                    {
                        "role": "assistant",
                        "content": (
                            f"Olá, {id_paciente}! 👋 A sua consulta de {tema_consulta} foi carregada.\n"
                            "Estou aqui para o ajudar a esclarecer qualquer dúvida sobre o que foi discutido.\n"
                            "Pode escrever a sua pergunta ou usar o microfone 🎤 aqui em baixo."
                        )
                    }
                ]

                st.session_state.todas_as_consultas[nome_da_sessao] = {
                    "executor": executor,
                    "historico_ia": [],
                    "mensagens_ecra": mensagens_iniciais
                }

                # Ativa a nova consulta no ecrã
                st.session_state.consulta_atual  = nome_da_sessao
                st.session_state.executor        = executor
                st.session_state.historico_ia    = []
                st.session_state.mensagens_ecra  = mensagens_iniciais
                st.session_state.resposta_audio  = None
                st.session_state.ultimo_audio_hash = None

            st.success("Consulta carregada! Já pode conversar com a Clara.")
        else:
            st.warning("Por favor, adicione um ficheiro de áudio no Passo 1.")

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.82rem; color:#7a96aa; text-align:center;'>"
        "Clara • Assistente de Saúde<br>"
        "As informações são apenas de apoio.<br>"
        "Consulte sempre o seu médico."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <script>
    function fixInputBackground() {
        const selectors = [
            '[data-testid="stBottomBlockContainer"]',
            '.stChatFloatingInputContainer',
            '.stChatInputContainer'
        ];
        selectors.forEach(sel => {
            document.querySelectorAll(sel).forEach(el => {
                el.style.background = '#f4f1ec';
                el.style.backgroundColor = '#f4f1ec';
                el.style.boxShadow = 'none';
                el.style.borderTop = 'none';
                Array.from(el.children).forEach(child => {
                    child.style.background = '#f4f1ec';
                    child.style.backgroundColor = '#f4f1ec';
                });
            });
        });
    }
    fixInputBackground();
    setInterval(fixInputBackground, 500);
    </script>
    """, unsafe_allow_html=True)


# ==================
# INTERFACE DE CHAT
# ==================
for msg in st.session_state.mensagens_ecra:
    avatar = "🩺" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Reproduz o último áudio da Clara automaticamente
if st.session_state.resposta_audio:
    st.audio(st.session_state.resposta_audio, format="audio/mp3", autoplay=True)


# ── Função central: processa qualquer pergunta (texto ou voz) ──
def processar_pergunta(pergunta: str):
    st.session_state.mensagens_ecra.append({"role": "user", "content": pergunta})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(pergunta)

    saudacoes_basicas = ['ola', 'olá', 'bom dia', 'boa tarde', 'boa noite', 'oi']

    if pergunta.lower().strip() in saudacoes_basicas:
        resposta = "Olá! 😊 Como posso ajudar a esclarecer as suas dúvidas hoje?"
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown(resposta)
        st.session_state.mensagens_ecra.append({"role": "assistant", "content": resposta})
        st.session_state.historico_ia.extend([HumanMessage(content=pergunta), AIMessage(content=resposta)])
        with st.spinner("A preparar a resposta em voz…"):
            st.session_state.resposta_audio = sintetizar_resposta(resposta)

    else:
        if st.session_state.executor is None:
            st.error(
                "Ainda não há nenhuma consulta carregada.\n"
                "Peça a um familiar ou cuidador que carregue a consulta na barra lateral, do lado esquerdo."
            )
            return

        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("A Clara está a consultar os seus documentos…"):
                resposta_agente = st.session_state.executor.invoke({
                    "input": pergunta,
                    "chat_history": st.session_state.historico_ia
                })
                texto_da_resposta = resposta_agente["output"]
                st.markdown(texto_da_resposta)

            with st.spinner("A preparar a resposta em voz…"):
                st.session_state.resposta_audio = sintetizar_resposta(texto_da_resposta)

        st.session_state.mensagens_ecra.append({"role": "assistant", "content": texto_da_resposta})
        st.session_state.historico_ia.extend([
            HumanMessage(content=pergunta),
            AIMessage(content=texto_da_resposta)
        ])

    # Sincroniza o histórico no arquivo de consultas
    if st.session_state.consulta_atual:
        st.session_state.todas_as_consultas[st.session_state.consulta_atual]["historico_ia"]   = st.session_state.historico_ia
        st.session_state.todas_as_consultas[st.session_state.consulta_atual]["mensagens_ecra"] = st.session_state.mensagens_ecra

    st.rerun()


# ── Entrada por VOZ e TEXTO ──
st.markdown("""
<style>
[data-testid="stAudioInput"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stAudioInput"] label { display: none !important; }
</style>
""", unsafe_allow_html=True)

audio_gravado = st.audio_input("🎤", label_visibility="collapsed")

if audio_gravado is not None:
    audio_bytes = audio_gravado.read()
    audio_hash  = hashlib.md5(audio_bytes).hexdigest()
    if st.session_state.ultimo_audio_hash != audio_hash:
        st.session_state.ultimo_audio_hash = audio_hash
        with st.spinner("A transcrever a sua pergunta…"):
            pergunta_voz = transcrever_pergunta(audio_bytes)
        if pergunta_voz:
            processar_pergunta(pergunta_voz)

if pergunta_texto := st.chat_input("Escreva aqui a sua pergunta…"):
    processar_pergunta(pergunta_texto)