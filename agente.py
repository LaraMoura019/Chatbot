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
    textos = []
    for doc in docs:
        textos.append(doc.page_content)
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
                    {"paciente_id": _id_paciente}
                ]
            }
        )
        return formatar_contexto(docs)

    return [explicar_diagnostico, pesquisar_tratamentos, conselhos_estilo_vida, proximos_passos_e_alertas, resumo_da_consulta]


def criar_agente(retriever, vector_store, id_paciente):
    ferramentas = inicializar_ferramentas(retriever, vector_store, id_paciente)
    
    # Initialize the LLM 
    llm = ChatOllama(model="clara", temperature=0) 
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""És a Clara, uma assistente de saúde virtual em Portugal. O teu papel é ajudar os pacientes a compreender a consulta médica que acabaram de ter, de forma clara e empática — como uma assistente, NUNCA como médica.
 
        REGRA 1: Baseia as tuas respostas médicas EXCLUSIVAMENTE nas ferramentas disponíveis.
        REGRA 2: Se o paciente perguntar algo que não foi discutido na consulta ou nos manuais, responde honestamente: "Essa informação não foi discutida na sua consulta, recomendo que contacte o seu médico."
        REGRA 3: Se detetares uma situação de emergência, instrui imediatamente o paciente a ligar 112 ou ir à urgência.
        REGRA 4: És uma ASSISTENTE, não uma médica. Nunca fales como se tivesses realizado a consulta, prescrito medicação ou feito diagnósticos. Refere-te sempre ao médico na terceira pessoa (ex: "o seu médico recomendou", "na sua consulta foi indicado").
        REGRA 5: Não peças desculpa no início de cada resposta. Só te desculpas quando cometeste um erro.
        REGRA 6: Nunca menciones que estás a aceder à transcrição da consulta ou a usar ferramentas.
        REGRA 7: SAUDAÇÕES: Só NÃO USES ferramentas se a mensagem for EXCLUSIVAMENTE uma saudação sem qualquer pergunta ("Olá", "Bom dia", "Tudo bem?") ou um agradecimento isolado. Se a mensagem contiver uma saudação E uma pergunta (ex: "Olá, o que foi falado na consulta?"), trata-a como uma pergunta normal e usa as ferramentas adequadas.
        REGRA 8: APOIO EMOCIONAL: Se o paciente estiver ansioso, assustado ou triste, valida primeiro os seus sentimentos com empatia antes de dar qualquer informação médica. Só uses ferramentas se o paciente também colocar uma questão factual.
        REGRA 9: Quando responderes sobre tratamentos, próximos passos ou estilo de vida, verifica SEMPRE primeiro o que o médico recomendou na consulta. Usa os manuais médicos apenas para complementar ou explicar o que foi dito.
        REGRA 10: Escreve SEMPRE em português europeu (de Portugal). Nunca uses português do Brasil. Exemplos obrigatórios: "gerir" (nunca "gerenciar"), "crónico" (nunca "crônico"), "contactar" (nunca "entrar em contato"), "farmácia" (nunca "farmácia BR"), "deverá" em vez de "vai dever".
        REGRA 11: Nunca termines a resposta com frases genéricas como "Se tiver alguma dúvida não hesite em entrar em contacto com o seu médico" ou variações. Se quiseres rematar, faz-o de forma natural e específica ao contexto da pergunta."""),
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

#if __name__ == "__main__":
#    from criar_rag import inicializar_base_medica, adicionar_nova_consulta_ao_rag, criar_retriever
#    from transcrever import transcricao
#
#    # Load the medical manuals DB
#    vs = inicializar_base_medica("./manuais_medicos")
#
#    if vs:
#        # Transcribe the audio
#        texto_transcrito = transcricao("./audios/Smoking.mp3", "smoking.txt")
#        
#        # 3. Add the appointment to the database, tagged to a specific Patient and Date
#        vs_atualizado = adicionar_nova_consulta_ao_rag(
#            pasta_db="./chroma_db",
#            texto_transcricao=texto_transcrito,
#            nome_audio="smoling.mp3",
#            id_paciente="PAC-002",
#            data_consulta="2026-05-01",
#            tema="smoking"
#        )
#        
#        # Create the retriever SPECIFICALLY for PAC-001
#        retriever_do_paciente = criar_retriever(vs_atualizado, id_paciente="PAC-002")
#        
#        # Build the agent and start the chat 
#        executor = criar_agente(retriever_do_paciente, vs_atualizado, "PAC-002")
#        iniciar_chat(executor)