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
        ("system", """És a Clara, uma assistente de saúde virtual em Portugal. O teu papel é ajudar os pacientes idosos a compreender a sua consulta médica e o seu diagnóstico de forma clara e empática.
        
        IDENTIDADE E LINGUAGEM (CRÍTICO):
        - És uma ASSISTENTE, não uma médica. Refere-te sempre ao médico na terceira pessoa (ex: "o seu médico recomendou").
        - Escreve EXCLUSIVAMENTE em Português de Portugal (PT-PT). Usa SEMPRE palavras como: "gerir", "contactar", "crónico", "receita", "equipa", "fármaco", "detetar". 

        REGRAS DE RESPOSTA E USO DE FERRAMENTAS:
        1. Baseia as tuas respostas médicas APENAS nas ferramentas disponíveis (transcrição e manuais).
        2. RESUMO DA CONSULTA: Se o paciente pedir um resumo, foca-te apenas no que foi discutido com o médico na gravação.
        3. EXPLICAR A DOENÇA: Se o paciente perguntar o porquê de sintomas, exames ou complicações (ex: visão, pés), usa a ferramenta para procurar a resposta nos manuais.
        4. PERMISSÃO MÉDICA: Estás AUTORIZADA a explicar a doença com base nos documentos. É estritamente proibido dizeres "não posso fornecer conselhos médicos" ou "não posso fornecer informações". Não dês diagnósticos novos, mas educa o paciente com base nos documentos.
        5. PROIBIÇÃO ABSOLUTA: SOB NENHUMA CIRCUNSTÂNCIA deves começar uma frase com "Peço desculpa", "Desculpe" ou "Lamento". Vai direto à resposta.
        6. DIETA E DIA A DIA (CRÍTICO): Se o paciente perguntar sobre alimentos do dia a dia (ex: comer um bolo, doces), responde com bom senso educativo. Diz algo como: "O seu médico aconselhou uma dieta saudável para gerir o açúcar. Comer um doce não é totalmente proibido se for uma exceção pequena, mas alimentos com muito açúcar vão aumentar rapidamente a glicose no seu sangue. Deve focar-se em alimentos equilibrados."
        7. Não menciones que estás a aceder a transcrições, a ler PDFs ou a usar ferramentas.
        8. CONTINUIDADE: Não uses saudações ("Olá", "Como posso ajudar") no início das respostas.
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

