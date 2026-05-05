import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import os 

# Importamos as funções que já criaste nos teus outros ficheiros!
# (Nota: substitui 'teu_ficheiro_do_agente' pelo nome real do teu ficheiro .py)
from agente import criar_agente 
from criar_rag import inicializar_base_medica, adicionar_nova_consulta_ao_rag, criar_retriever
from transcrever import transcricao

# Configuração da página Web
st.set_page_config(page_title="Assistente Médico AI", page_icon="🩺")
st.title("🩺 Assistente Médico Virtual")

# ==========================================
# 1. GESTÃO DE MEMÓRIA (SESSION STATE)
# ==========================================
# Guardamos o histórico para a IA ler
if "historico_ia" not in st.session_state:
    st.session_state.historico_ia = []

# Guardamos as mensagens para mostrar no ecrã bonito do Streamlit
if "mensagens_ecra" not in st.session_state:
    st.session_state.mensagens_ecra = [
        {"role": "assistant", "content": "Olá! Sou o teu Assistente de Saúde. Como posso ajudar hoje?"}
    ]

# Guardamos o executor (o agente) para não ter de ser recriado a cada clique
if "executor" not in st.session_state:
    st.session_state.executor = None


# ==========================================
# 2. BARRA LATERAL (CONFIGURAÇÃO)
# ==========================================
with st.sidebar:
    st.header("⚙️ Painel de Controlo")
    st.write("Adicione a nova consulta ao sistema.")
    
    # 1. A Caixa mágica de Upload!
    ficheiro_audio = st.file_uploader("Upload do áudio da consulta", type=["mp3", "wav", "m4a"])
    
    # 2. Informações para a Metadata do RAG
    id_paciente = st.text_input("ID do Paciente", value="PAC-001")
    tema_consulta = st.text_input("Tema (ex: diabetes, fumar)", value="Geral")
    data_consulta = st.date_input("Data da Consulta")
    
    # 3. Botão para processar tudo
    if st.button("Carregar Consulta"):
        if ficheiro_audio is not None:
            with st.spinner("A processar áudio e a atualizar o cérebro da IA..."):
                
                # Criar a pasta audios se ela não existir
                os.makedirs("./audios", exist_ok=True)
                
                # Guardar o ficheiro que o utilizador enviou no disco do computador
                caminho_audio_temp = f"./audios/{ficheiro_audio.name}"
                with open(caminho_audio_temp, "wb") as f:
                    f.write(ficheiro_audio.getbuffer())
                
                # ---- O TEU CÓDIGO RAG ENTRA AQUI ----
                vs = inicializar_base_medica("./manuais_medicos")
                
                # Opcional: damos um nome ao ficheiro de texto baseado no nome do áudio
                nome_txt = ficheiro_audio.name.replace(".", "_") + ".txt"
                
                # Transcreve o ficheiro que acabou de ser guardado!
                texto_transcrito = transcricao(caminho_audio_temp, nome_txt)
                
                vs_atualizado = adicionar_nova_consulta_ao_rag(
                    pasta_db="./chroma_db",
                    texto_transcricao=texto_transcrito,
                    nome_audio=ficheiro_audio.name,
                    id_paciente=id_paciente,
                    data_consulta=str(data_consulta), # O Streamlit usa o formato de data correto
                    tema=tema_consulta
                )
                
                retriever_do_paciente = criar_retriever(vs_atualizado, id_paciente=id_paciente)
                
                # Guardamos o agente na memória da página!
                st.session_state.executor = criar_agente(retriever_do_paciente, vs_atualizado, id_paciente)
                
            st.success("✅ Base de dados e Agente prontos! Já pode falar.")
        else:
            st.error("⚠️ Por favor, faça o upload de um ficheiro de áudio antes de clicar no botão.")


# ==========================================
# 3. INTERFACE DE CHAT (O ECRÃ PRINCIPAL)
# ==========================================
# ... (o resto do teu código do chat fica exatamente igual) ...

# ==========================================
# 3. INTERFACE DE CHAT (O ECRÃ PRINCIPAL)
# ==========================================

# Mostra todas as mensagens anteriores no ecrã
for msg in st.session_state.mensagens_ecra:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# A caixa de texto onde o utilizador escreve
if pergunta := st.chat_input("Escreva aqui a sua dúvida..."):
    
    # 1. Mostra a pergunta do utilizador no ecrã
    st.session_state.mensagens_ecra.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    # 2. O NOSSO "ESCUDO" DE PYTHON PARA SAUDAÇÕES
    saudacoes_basicas = ['ola', 'olá', 'bom dia', 'boa tarde', 'boa noite', 'oi']
    if pergunta.lower().strip() in saudacoes_basicas:
        resposta = "Olá! Como te posso ajudar com as dúvidas sobre a tua consulta hoje?"
        
        # Mostra a resposta e guarda no histórico
        with st.chat_message("assistant"):
            st.markdown(resposta)
        st.session_state.mensagens_ecra.append({"role": "assistant", "content": resposta})
        st.session_state.historico_ia.extend([HumanMessage(content=pergunta), AIMessage(content=resposta)])

    # 3. SE NÃO FOR UMA SAUDAÇÃO, CHAMA O AGENTE (A IA)
    else:
        if st.session_state.executor is None:
            st.error("⚠️ Por favor, carrega a consulta na barra lateral primeiro!")
        else:
            with st.chat_message("assistant"):
                # O spinner mostra aquela animação de "A pensar..."
                with st.spinner("A consultar os manuais e a transcrição..."):
                    
                    # Pede a resposta ao agente
                    resposta_agente = st.session_state.executor.invoke({
                        "input": pergunta,
                        "chat_history": st.session_state.historico_ia
                    })
                    
                    texto_da_resposta = resposta_agente["output"]
                    st.markdown(texto_da_resposta)
            
            # Guarda na memória do ecrã e na memória da IA
            st.session_state.mensagens_ecra.append({"role": "assistant", "content": texto_da_resposta})
            st.session_state.historico_ia.extend([
                HumanMessage(content=pergunta),
                AIMessage(content=texto_da_resposta)
            ])