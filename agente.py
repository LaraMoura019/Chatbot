from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage

# Variáveis globais para guardar a base de dados e o ID
_retriever = None
_vector_store = None
_id_paciente = None

def formatar_contexto(docs):
    """
    Takes the chunks of text found by the retriever and joins them 
    into a single, clean text separated by paragraphs for easier reading.
    """
    textos = []
    for doc in docs:
        textos.append(doc.page_content)
    
    # Join all texts with double line breaks between them
    return "\n\n".join(textos)


def inicializar_ferramentas(retriever, vector_store, id_paciente):
    global _retriever, _vector_store, _id_paciente
    _retriever = retriever
    _vector_store = vector_store
    _id_paciente = id_paciente

    @tool
    def explicar_diagnostico(pergunta: str) -> str:
        """
        Use this tool to explain diagnoses, diseases, causes of health problems, 
        and the reasons behind the patient's symptoms.
        """
        docs = _retriever.invoke(pergunta + " diagnóstico explicação sintomas causa")
        return formatar_contexto(docs)

    @tool
    def pesquisar_tratamentos(pergunta: str) -> str:
        """
        Use this tool for questions regarding treatments, medications, 
        pills, dosages, medical prescriptions, side effects, or medical exams.
        """
        docs = _retriever.invoke(pergunta + " medicação tratamento dose exames receita")
        return formatar_contexto(docs)

    @tool
    def conselhos_estilo_vida(pergunta: str) -> str:
        """
        Use this tool for questions about daily life and habits: nutrition, 
        diet, physical exercise, sleep, posture, and stress management.
        """
        docs = _retriever.invoke(pergunta + " hábitos alimentação exercício recomendações")
        return formatar_contexto(docs)

    @tool
    def proximos_passos_e_alertas(pergunta: str) -> str:
        """
        Use this tool to find out when the patient should return to the doctor, 
        what the next steps are, or what the warning signs/emergency triggers are.
        """
        docs = _retriever.invoke(pergunta + " próxima consulta emergência urgência perigo atenção")
        return formatar_contexto(docs)

    @tool
    def resumo_da_consulta(pergunta: str) -> str:
        """
        Use this tool when the user asks for a general summary of the appointment, 
        what was discussed, what the patient felt, or what the doctor said.
        """
        # Vai diretamente à base de dados buscar APENAS os textos da consulta deste paciente
        docs = _vector_store.similarity_search(
            "resumo da consulta sintomas médico paciente",
            k=10, # Traz até 10 blocos de texto da consulta
            filter={
                "$and": [
                    {"tipo": "consulta_medica"},
                    {"paciente_id": _id_paciente}
                ]
            }
        )
        return formatar_contexto(docs)

    return [explicar_diagnostico, pesquisar_tratamentos, conselhos_estilo_vida, proximos_passos_e_alertas, resumo_da_consulta]


def criar_agente(retriever, vector_store, id_paciente):
    # Agora passamos os 3 argumentos para as ferramentas
    ferramentas = inicializar_ferramentas(retriever, vector_store, id_paciente)
    
    # Initialize the LLM (the brain)
    llm = ChatOllama(model="llama3.1:8b", temperature=0) 
    
    # The Prompt defines the personality and strict rules of our bot
    prompt = ChatPromptTemplate.from_messages([
        # Base rules
        ("system", """You are a generalist virtual medical assistant in Portugal.
        Your job is to help patients understand the medical appointment they just had, regardless of the specialty.

        RULE 1: Base your answers EXCLUSIVELY on the provided tools (appointment transcription and manuals).
        RULE 2: If the patient asks about something that was not discussed in the appointment or is not in your manuals, state honestly: "Essa informação não foi discutida na sua consulta, recomendo que contacte o seu médico."
        RULE 3: If you detect any emergency situation, immediately advise contacting 112 or going to the emergency room.
        RULE 4: Maintain a welcoming tone and never try to replace the human doctor. Respond in European Portuguese.
        RULE 5: Do not apologize every time you start a sentence, only when you make a mistake!
        RULE 6: Never mention that you are accessing the appointment transcription. If the patient asks something about the appointment, simply use the data you have regarding the transcription without explaining the process."""),
        
        # Store past conversation history
        MessagesPlaceholder(variable_name="chat_history"),
        
        # The user's new question
        ("human", "{input}"),
        
        # Blank space where the agent "thinks" and calls tools
        ("placeholder", "{agent_scratchpad}"), 
    ])
    
    # Create the agent by combining the LLM, tools, and prompt
    agente = create_tool_calling_agent(llm, ferramentas, prompt)
    
    # The Executor puts the agent to work and handles errors
    executor = AgentExecutor(
        agent=agente,
        tools=ferramentas,
        verbose=True, # Set to True to see the "thinking" steps in the terminal
        handle_parsing_errors=True
    )
    
    return executor


# ─────────────────────────────────────────────
# 4. CHAT LOOP
# ─────────────────────────────────────────────
def iniciar_chat(executor):
    print("\nOlá! Sou o teu Assistente de Saúde. Como te posso ajudar hoje?")
    print("(Escreve 'sair' para terminar a conversa)\n")
    
    # This list will store the conversation memory
    historico_conversa = []
    
    while True:
        pergunta = input("Tu: ")
        
        if pergunta.lower() == 'sair':
            print("As melhoras! Até à próxima.")
            break
            
        try:
            # Send the question AND the memory to the Agent
            resposta = executor.invoke({
                "input": pergunta,
                "chat_history": historico_conversa
            })
            
            texto_da_resposta = resposta["output"]
            print(f"\nAssistente: {texto_da_resposta}\n")
            
            # Save this interaction to the memory
            historico_conversa.extend([
                HumanMessage(content=pergunta),
                AIMessage(content=texto_da_resposta)
            ])
            
        except Exception as e:
            print(f"Ups, houve um erro: {e}")

# ─────────────────────────────────────────────
# HOW TO RUN THE FULL PIPELINE:
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from criar_rag import inicializar_base_medica, adicionar_nova_consulta_ao_rag, criar_retriever
    from transcrever import transcricao

    # 1. Load the medical manuals DB (fast if it already exists)
    vs = inicializar_base_medica("./manuais_medicos")

    if vs:
        # 2. Transcribe the audio
        texto_transcrito = transcricao("./audios/diabetes.mp3", "diabetes.txt")
        
        # 3. Add the appointment to the database, tagged to a specific Patient and Date
        vs_atualizado = adicionar_nova_consulta_ao_rag(
            pasta_db="./chroma_db",
            texto_transcricao=texto_transcrito,
            nome_audio="diabetes.mp3",
            id_paciente="PAC-001",
            data_consulta="2026-05-01",
            tema="diabetes"
        )
        
        # 4. Create the retriever SPECIFICALLY for PAC-001
        retriever_do_paciente = criar_retriever(vs_atualizado, id_paciente="PAC-001")
        
        # 5. Build the agent and start the chat (Passing all 3 arguments!)
        executor = criar_agente(retriever_do_paciente, vs_atualizado, "PAC-001")
        iniciar_chat(executor)