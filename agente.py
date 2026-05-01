from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage

# Variável global para guardar o  motor de busca
_retriever = None

def formatar_contexto(docs):
    """
    Esta função pega nos pedaços de texto (chunks) que o retriever encontrou
    e junta-os todos num único texto limpo, separado por parágrafos.
    Isto facilita a leitura.
    """
    textos = []
    for doc in docs:
        textos.append(doc.page_content)
    
    # Junta os textos todos com duas quebras de linha entre eles
    return "\n\n".join(textos)


# Passamos a usar a vector_store em vez do retriever genérico
def inicializar_ferramentas(vector_store):

    @tool
    def ler_resumo_da_consulta(pergunta: str) -> str:
        """
        PASSO 1: Usa SEMPRE esta ferramenta PRIMEIRO para descobrir o que 
        foi falado na consulta e qual é a doença ou sintomas do paciente.
        """
        docs = vector_store.similarity_search(
            pergunta, k=3, filter={"tipo": "consulta_medica"} # Só lê o áudio!
        )
        return formatar_contexto(docs)

    @tool
    def explicar_diagnostico(doenca_ou_sintoma: str) -> str:
        """
        PASSO 2: Usa esta ferramenta para explicar a doença DO PACIENTE.
        Obrigatório: Passa o nome da doença (ex: 'diabetes') e não a pergunta toda.
        """
        # Procura apenas nos PDFs médicos, focando na doença específica
        docs = vector_store.similarity_search(
            doenca_ou_sintoma + " diagnóstico explicação sintomas causa", 
            k=3, filter={"tipo": "conhecimento_medico"}
        )
        return formatar_contexto(docs)

    @tool
    def pesquisar_tratamentos(doenca_ou_sintoma: str) -> str:
        """
        PASSO 2: Usa para ver tratamentos e medicamentos para a doença DO PACIENTE.
        Obrigatório: Passa o nome da doença (ex: 'diabetes').
        """
        docs = vector_store.similarity_search(
            doenca_ou_sintoma + " medicação tratamento dose exames receita", 
            k=3, filter={"tipo": "conhecimento_medico"}
        )
        return formatar_contexto(docs)

    @tool
    def conselhos_estilo_vida(doenca_ou_sintoma: str) -> str:
        """
        PASSO 2: Usa para ver hábitos e alimentação para a doença DO PACIENTE.
        Obrigatório: Passa o nome da doença (ex: 'diabetes').
        """
        docs = vector_store.similarity_search(
            doenca_ou_sintoma + " hábitos alimentação exercício recomendações", 
            k=3, filter={"tipo": "conhecimento_medico"}
        )
        return formatar_contexto(docs)

    @tool
    def proximos_passos_e_alertas(doenca_ou_sintoma: str) -> str:
        """
        PASSO 2: Usa para ver sinais de perigo da doença DO PACIENTE.
        Obrigatório: Passa o nome da doença (ex: 'diabetes').
        """
        docs = vector_store.similarity_search(
            doenca_ou_sintoma + " próxima consulta emergência urgência perigo", 
            k=3, filter={"tipo": "conhecimento_medico"}
        )
        return formatar_contexto(docs)

    return [ler_resumo_da_consulta, explicar_diagnostico, pesquisar_tratamentos, conselhos_estilo_vida, proximos_passos_e_alertas]


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
# COMO CORRER ISTO TUDO NO FINAL:
# ─────────────────────────────────────────────
from criar_rag import inicializar_rag
from transcrever import transcricao
retriever, vs = inicializar_rag("manuais_medicos",transcricao("./audios/diabetes.mp3","diabetes.txt"),"diabetes.mp3")
executor = criar_agente(retriever)
iniciar_chat(executor)