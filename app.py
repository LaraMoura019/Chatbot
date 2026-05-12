import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agente import criar_agente
from criar_rag import inicializar_base_medica, adicionar_nova_consulta_ao_rag, criar_retriever
from transcrever import transcricao

st.set_page_config(
    page_title="Clara – Assistente de Saúde",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

/* ── Fundo ── */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #f4f1ec 0%, #eef3f8 60%, #f0ece4 100%) !important;
    font-family: 'Nunito', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Título ── */
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

/* ── Texto dos balões — LETRA GRANDE para seniores ── */
[data-testid="stChatMessageContent"] p {
    font-family: 'Nunito', sans-serif !important;
    font-size: 1.08rem !important;
    line-height: 1.8 !important;
    color: #1e2d3d !important;
}

/* Balão do utilizador — azul acinzentado */
[data-testid="stChatMessage"][data-author="user"] [data-testid="stChatMessageContent"] {
    background: #ddeaf5 !important;
    border-radius: 20px 20px 6px 20px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(45,74,107,0.10);
    border: 1.5px solid #bdd1e6;
}

/* Balão da Clara — creme quente */
[data-testid="stChatMessage"][data-author="assistant"] [data-testid="stChatMessageContent"] {
    background: #fdf7ee !important;
    border-radius: 20px 20px 20px 6px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(180,150,100,0.10);
    border: 1.5px solid #e8d9bf;
}

/* ── Área inferior do chat (fundo igual ao resto) ── */
[data-testid="stBottomBlockContainer"],
[data-testid="stBottomBlockContainer"] > div,
[data-testid="stBottomBlockContainer"] > div > div,
[data-testid="stChatInput"],
section[data-testid="stChatInput"],
.stChatFloatingInputContainer,
.stChatFloatingInputContainer > div,
.stChatInputContainer {
    background: #f4f1ec !important;
    background-color: #E8F0F7!important;
    box-shadow: none !important;
    border-top: none !important;
}

/* ── Caixa de input — grande e confortável ── */
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

/* ── Sidebar ── */
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

/* Info box */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: #e8f4fd !important;
    border: 1.5px solid #b5d3ea !important;
    border-radius: 12px !important;
    color: #2d4a6b !important;
    font-size: 0.97rem !important;
}

/* Inputs sidebar */
[data-testid="stSidebar"] input {
    border-radius: 10px !important;
    border: 2px solid #bdd1e6 !important;
    background: #f8fbff !important;
    color: #1e2d3d !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 1rem !important;
    padding: 8px 12px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2.5px dashed #7aa5c8 !important;
    border-radius: 14px !important;
    background: #f0f7fd !important;
    padding: 10px !important;
}

/* Botão — grande, alto contraste, fácil de carregar */
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

/* Ocultar menu e rodapé do Streamlit */
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
if "historico_ia" not in st.session_state:
    st.session_state.historico_ia = []

if "mensagens_ecra" not in st.session_state:
    st.session_state.mensagens_ecra = [
        {
            "role": "assistant",
            "content": (
                "Olá! 👋 Sou a **Clara**, a sua assistente de saúde.\n"
                "Estou aqui para o ajudar a esclarecer as dúvidas sobre a sua consulta.\n"
                "Escreva a sua pergunta aqui em baixo, com calma, e eu respondo."
            )
        }
    ]

if "executor" not in st.session_state:
    st.session_state.executor = None


# ================
# BARRA LATERAL
# ================
with st.sidebar:
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
                texto_transcrito = transcricao(caminho_audio_temp, nome_txt)

                vs_atualizado = adicionar_nova_consulta_ao_rag(
                    pasta_db="./chroma_db",
                    texto_transcricao=texto_transcrito,
                    nome_audio=ficheiro_audio.name,
                    id_paciente=id_paciente,
                    data_consulta=str(data_consulta),
                    tema=tema_consulta
                )

                retriever_do_paciente = criar_retriever(vs_atualizado, id_paciente=id_paciente)
                st.session_state.executor = criar_agente(retriever_do_paciente, vs_atualizado, id_paciente)

                # Reinicia o chat para a nova consulta
                st.session_state.historico_ia = []
                st.session_state.mensagens_ecra = [
                    {
                        "role": "assistant",
                        "content": (
                            f"Olá, {id_paciente}! 👋 A sua consulta foi carregada com sucesso.\n"
                            "Estou aqui para o ajudar a esclarecer qualquer dúvida sobre o que foi discutido.\n"
                            "Escreva a sua pergunta aqui em baixo, com calma, e eu respondo."
                        )
                    }
                ]

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
                // Apanha também todos os filhos diretos
                Array.from(el.children).forEach(child => {
                    child.style.background = '#f4f1ec';
                    child.style.backgroundColor = '#f4f1ec';
                });
            });
        });
    }
    // Corre ao carregar e de 500 em 500ms para apanhar elementos carregados depois
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

if pergunta := st.chat_input("Escreva aqui a sua pergunta e prima Enter…"):

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

    else:
        if st.session_state.executor is None:
            st.error(
                "Ainda não há nenhuma consulta carregada.\n"
                "Peça a um familiar ou cuidador que carregue a consulta na barra lateral, do lado esquerdo."
            )
        else:
            with st.chat_message("assistant", avatar="🩺"):
                with st.spinner("A Clara está a consultar os seus documentos…"):
                    resposta_agente = st.session_state.executor.invoke({
                        "input": pergunta,
                        "chat_history": st.session_state.historico_ia
                    })
                    texto_da_resposta = resposta_agente["output"]
                    st.markdown(texto_da_resposta)

            st.session_state.mensagens_ecra.append({"role": "assistant", "content": texto_da_resposta})
            st.session_state.historico_ia.extend([
                HumanMessage(content=pergunta),
                AIMessage(content=texto_da_resposta)
            ])