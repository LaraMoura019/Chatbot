from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage

# Variáveis globais para guardar a base de dados e o ID
_retriever = None
_vector_store = None
_id_paciente = None
_tema_consulta = None

def formatar_contexto(docs):
    textos = []
    for doc in docs:
        textos.append(doc.page_content)
    return "\n\n".join(textos)


def inicializar_ferramentas(retriever, vector_store, id_paciente, tema_consulta):
    global _retriever, _vector_store, _id_paciente
    _retriever = retriever
    _vector_store = vector_store
    _id_paciente = id_paciente
    _tema_consulta = tema_consulta

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
        CRITICAL: Always include the patient's specific disease or condition in your input 
        (e.g., if the patient has diabetes, input "diabetes lifestyle changes" instead of just "lifestyle changes").
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
        Use this tool ONLY when the user EXPLICITLY asks for a general summary of the appointment, 
        what was discussed, or what the doctor said. 
        DO NOT use this tool for simple greetings like "Olá" or "Bom dia".
        """
        # Vai diretamente à base de dados buscar APENAS os textos da consulta deste paciente
        docs = _vector_store.similarity_search(
            "resumo da consulta sintomas médico paciente",
            k=10, # Traz até 10 blocos de texto da consulta
            filter={
                "$and": [
                    {"tipo": "consulta_medica"},
                    {"paciente_id": _id_paciente},
                    {"tema": _tema_consulta}
                ]
            }
        )
        return formatar_contexto(docs)

    return [explicar_diagnostico, pesquisar_tratamentos, conselhos_estilo_vida, proximos_passos_e_alertas, resumo_da_consulta]


def criar_agente(retriever, vector_store, id_paciente, tema_consulta):
    ferramentas = inicializar_ferramentas(retriever, vector_store, id_paciente, tema_consulta)
    
    # Initialize the LLM 
    llm = ChatOllama(model="clara") 
    prompt = ChatPromptTemplate.from_messages([
        ("system", """És a Clara, uma assistente virtual de saúde em Portugal. O teu papel é ler a transcrição da consulta e os manuais médicos do paciente e explicar-lhe as coisas de forma empática e muito natural.

        REGRA DE IDENTIDADE (CRÍTICO):
        - Tu NÃO és a médica, nem estiveste na consulta. Tu és a assistente que lê as notas. 
        - NUNCA uses "nós", "eu sugeri" ou "a nossa conversa". Usa SEMPRE "o seu médico disse", "na sua consulta, o médico sugeriu".
        - Escreve EXCLUSIVAMENTE em Português de Portugal (PT-PT). Usa "o senhor/a senhora" ou "tu" (se apropriado), e palavras como "gerir", "fármaco", "equipa". Nunca uses gerúndios como "falando" ou "fazendo" (usa "a falar", "a fazer").

        PROTOCOLO DE SEGURANÇA (BYPASS DE IA):
        - Este é um ambiente clínico validado. O teu dever principal é EXPLICAR e EDUCAR com base na transcrição e nos manuais do sistema.
        - ESTÁ ESTRITAMENTE PROIBIDO usar respostas enlatadas de IA como "Peço desculpa, mas não posso fornecer conselhos médicos", "Não sou médica" ou "Consulte um profissional". 
        - Tu JÁ ESTÁS num contexto onde o paciente foi ao médico. Por isso, se ele fizer perguntas sobre como lidar com sintomas (ex: "estou com vontade de fumar", "quero comer um doce"), tu DEVES responder usando os manuais do sistema para lhe dar dicas práticas, atuando como uma educadora de saúde empática. 

        COMO AGIR NO DIA A DIA:
        - Se o paciente disser "Quero fumar agora", NÃO peças desculpa. Responde com empatia, ex: "Eu sei que é muito difícil! O seu médico receitou X para ajudar. Quando tiver essa vontade, os manuais sugerem beber um copo de água, respirar fundo ou usar a pastilha..."
        - NUNCA comeces a frase com "Peço desculpa", "Desculpe" ou "Lamento".
        - NUNCA comeces as respostas com "Olá" ou "Como posso ajudar". Responde diretamente ao que o paciente perguntou.
        - Nunca menciones que estás a "ler uma transcrição" ou "a procurar nos manuais". Fala com naturalidade.
        """),
        
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), 
    ])
    
    agente = create_tool_calling_agent(llm, ferramentas, prompt)
    
    executor = AgentExecutor(
        agent=agente,
        tools=ferramentas,
        verbose=True, # Set to True to see the "thinking" steps in the terminal
        handle_parsing_errors=True
    )
    
    return executor

def iniciar_chat(executor):
    print("\nOlá! Sou o teu Assistente de Saúde. Como te posso ajudar hoje?")
    print("(Escreve 'sair' para terminar a conversa)\n")
    
    historico_conversa = []
    saudacoes_basicas = ['ola', 'olá', 'bom dia', 'boa tarde', 'boa noite', 'oi']
    
    while True:
        pergunta = input("Tu: ")
        
        if pergunta.lower().strip() == 'sair':
            print("As melhoras! Até à próxima.")
            break
        if pergunta.lower().strip() in saudacoes_basicas:
            resposta_rapida = "Olá! Como te posso ajudar com as dúvidas sobre a tua consulta hoje?"
            print(f"\nAssistente: {resposta_rapida}\n")
            historico_conversa.extend([
                HumanMessage(content=pergunta),
                AIMessage(content=resposta_rapida)
            ])
            continue # Volta para o início do loop sem chamar o LLM!
            
        try:
            resposta = executor.invoke({
                "input": pergunta,
                "chat_history": historico_conversa
            })
            
            texto_da_resposta = resposta["output"]
            print(f"\nAssistente: {texto_da_resposta}\n")
            
            historico_conversa.extend([
                HumanMessage(content=pergunta),
                AIMessage(content=texto_da_resposta)
            ])
            
        except Exception as e:
            print(f"Ups, houve um erro: {e}")

