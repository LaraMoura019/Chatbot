from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage

# Global variable to store the search engine (retriever)
_retriever = None

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


def inicializar_ferramentas(retriever):
    global _retriever
    _retriever = retriever

    @tool
    def resumo_da_consulta(pergunta: str) -> str:
        """
        Use this tool when the user asks for a general summary of the appointment, 
        what was discussed, what the patient felt, or what the doctor said.
        """
        # Injetamos palavras que tipicamente existem numa transcrição médica
        docs = _retriever.invoke("paciente relata médico queixas recomenda sintomas história clínica")
        return formatar_contexto(docs)

    @tool
    def explicar_diagnostico(pergunta: str) -> str:
        """
        Use this tool to explain diagnoses, diseases, causes of health problems, 
        and the reasons behind the patient's symptoms.
        """
        # Appending Portuguese keywords to help the semantic search
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

    return [explicar_diagnostico, pesquisar_tratamentos, conselhos_estilo_vida, proximos_passos_e_alertas, resumo_da_consulta]


def criar_agente(retriever):
    ferramentas = inicializar_ferramentas(retriever)
    
    # Inicializamos o cérebro (LLM)
    llm = ChatOllama(model="llama3.1:8b", temperature=0) 
    
    # O Prompt é a "Personalidade" do nosso bot e as suas regras
    prompt = ChatPromptTemplate.from_messages([
        # Regra base
        ("system", """You are a generalist virtual medical assistant in Portugal.
        Your job is to help patients understand the medical appointment they just had, regardless of the specialty.

        RULE 1: Base your answers EXCLUSIVELY on the provided tools (appointment transcription and manuals).
        RULE 2: If the patient asks about something that was not discussed in the appointment or is not in your manuals, state honestly: "Essa informação não foi discutida na sua consulta, recomendo que contacte o seu médico."
        RULE 3: If you detect any emergency situation, immediately advise contacting 112 or going to the emergency room.
        RULE 4: Maintain a welcoming tone and never try to replace the human doctor. Respond in European Portuguese.
        RULE 5: Do not apologize every time you start a sentence, only when you make a mistake!
        RULE 6: Never mention that you are accessing the appointment transcription. If the patient asks something about the appointment, simply use the data you have regarding the transcription without explaining the process."""),
        # guardamos a conversa passada.
        MessagesPlaceholder(variable_name="chat_history"),
        
        # A nova pergunta do utilizador
        ("human", "{input}"),
        
        # Espaço em branco onde o agente "pensa" e usa as ferramentas
        ("placeholder", "{agent_scratchpad}"), 
    ])
    
    # Criamos o agente juntando o cérebro (llm), os olhos (ferramentas) e a personalidade (prompt)
    agente = create_tool_calling_agent(llm, ferramentas, prompt)
    
    # O Executor é quem põe o agente a trabalhar na prática
    executor = AgentExecutor(
        agent=agente,
        tools=ferramentas,
        verbose=True, # Ligado para vermos os passos no ecrã (ótimo para aprenderes!)
        handle_parsing_errors=True # Proteção caso a IA se engane no formato
    )
    
    return executor


# ─────────────────────────────────────────────
# 4. INICIAR A CONVERSA (O Loop do Chatbot)
# ─────────────────────────────────────────────
def iniciar_chat(executor):
    print("\nOlá! Sou o teu Assistente de Saúde. Como te posso ajudar hoje?")
    print("(Escreve 'sair' para terminar a conversa)\n")
    
    # Esta lista vai guardar tudo o que vocês disserem (A Memória!)
    historico_conversa = []
    
    while True:
        pergunta = input("Tu: ")
        
        if pergunta.lower() == 'sair':
            print("As melhoras! Até à próxima.")
            break
            
        try:
            # Mandamos a pergunta E a memória para o Agente
            resposta = executor.invoke({
                "input": pergunta,
                "chat_history": historico_conversa
            })
            
            texto_da_resposta = resposta["output"]
            print(f"\nAssistente: {texto_da_resposta}\n")
            
            # Depois de responder, guardamos esta troca de mensagens na memória
            historico_conversa.extend([
                HumanMessage(content=pergunta),
                AIMessage(content=texto_da_resposta)
            ])
            
        except Exception as e:
            print(f"Ups, houve um erro: {e}")

# ─────────────────────────────────────────────
# COMO CORRER ISTO TUDO NO FINAL (ATUALIZADO):
# ─────────────────────────────────────────────
# Importamos as funções novas do ficheiro RAG
from criar_rag import inicializar_base_medica, adicionar_nova_consulta_ao_rag, criar_retriever
from transcrever import transcricao

# 1. Carrega a base de dados com os manuais médicos (rápido se já existir)
vs = inicializar_base_medica("./manuais_medicos")

if vs:
    # 2. Fazemos a transcrição do áudio
    texto_transcrito = transcricao("./audios/diabetes.mp3", "diabetes.txt")
    
    # 3. Guardamos a consulta no banco de dados, associando a um Paciente e a uma Data!
    vs_atualizado = adicionar_nova_consulta_ao_rag(
        pasta_db="./chroma_db",
        texto_transcricao=texto_transcrito,
        nome_audio="diabetes.mp3",
        id_paciente="PAC-001",
        data_consulta="2026-05-01",
        tema="diabetes"
    )
    
    # 4. Criamos o retriever ESPECÍFICO para o PAC-001
    # Assim o Agente só vai ler os PDFs médicos e as consultas deste paciente específico
    retriever_do_paciente = criar_retriever(vs_atualizado, id_paciente="PAC-001")
    
    # 5. Entregamos os "olhos" certos ao Agente e iniciamos o chat!
    executor = criar_agente(retriever_do_paciente)
    iniciar_chat(executor)