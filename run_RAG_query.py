from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import random
import pickle
import time

load_dotenv()

def answer_queries(response_file):
    vectorstore= FAISS.load_local("pdf_vector_store", embeddings=OpenAIEmbeddings(api_key=os.getenv("API_KEY"), model="text-embedding-3-large"), allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    #llm = ChatOpenAI(model="gpt-5", api_key=os.getenv("API_KEY"))
    llm = ChatOllama(model="gemma3:27b", temperature=0.3)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieve_and_format(question):
        docs = retriever.invoke(question)
        return {"context": format_docs(docs), "question": question, "source_documents": docs}
    
    main = """
                The question will describe and instrument and ask if the instrument is capable of performing a measurement.
                If one of the main purposes of this instrument is to perform the measurement we will categorize it as "Primary".
                If this instrument can perform the measurement but is not best suited for the job or is not commonly used, we will categorize it as "Secondary".
                If this instrument cannot be used to perform the measurement, we will categorize it as "Tertiary".
                Answer the question by responding with the appropriate category label for the instrument with an explanation of your response afterwards given your prior knowledge and the context provided.
                Give a confidence score from 0 to 100 based on how certain you are about the answer you are providing.
            """
    prompt = ChatPromptTemplate.from_template(main+"""
                                                                                      
    Context:
    {context}

    Question: {question}
                                              """)

    rag_chain = (
        RunnableLambda(retrieve_and_format)
        | RunnablePassthrough.assign(
            answer=prompt | llm | StrOutputParser()
        )
    )

    responses = {}

    try:
        with open(f"responses/{response_file}.pkl", "rb") as file:
            responses = pickle.load(file)

    except FileNotFoundError:
        print(f"No current {response_file}.pkl in responses/ folder available to load.")

    answered_queries = responses.keys()
    with open("queries.txt", "r", encoding='utf-8') as file:
        content = file.read()
    
    lines = content.split("A sensor is of type")
    lines = random.sample(lines, 100)

    start = time.time()
    for line in lines:
        inter_start = time.time()
        if inter_start - start > 3600:
            print("TIMEOUT - started queries at {start}, this query started at {inter_start}")
            break

        query="A sensor is of type "+line.strip()
        messages = [("system","You are an Earth Science expert validating the capabilities of measurement devices on Earth Observation satellites."),
                    ("human",f"{main}\n Question:{query}"),]

        if query not in answered_queries:
            rag_result = rag_chain.invoke(query)
            llm_result = llm.invoke(messages)
            responses[query] = {"answer":rag_result["answer"], "context":rag_result["context"], "llm":llm_result.text}
        elif "llm" not in responses[query].keys():
            llm_result = llm.invoke(messages)
            responses[query]["llm"] = llm_result.text
        
        print(f"{time.time()-inter_start}\n")


    with open("responses.pkl", "wb") as file:
        pickle.dump(responses,file)


def read_responses(response_file):
    responses = {}
    with open(f"responses/{response_file}.pkl", "rb") as file:
        responses = pickle.load(file)

    incorrect_rag = 0
    incorrect_llm = 0
    total = 0
    for query in responses.keys():
        if "llm" in responses[query].keys():
            total += 1
            if "tertiary" in responses[query]["answer"].lower()[:10]:
                incorrect_rag += 1
                print(f"Query - {query}")
                print(f"RAG Response - {responses[query]['answer']}")
                #print(responses[query]["context"])
            
            if "tertiary" in responses[query]["llm"].lower()[:10]:
                incorrect_llm += 1
                print(f"LLM response - {responses[query]['llm']}")
    
    print(f"Incorrect RAG responses = {incorrect_rag}/{total}")
    print(f"Incorrect Zero-shot responses = {incorrect_llm}/{total}")


if __name__ == "__main__":
    response_file = "responses_gpt-5"
    #answer_queries(response_file)
    read_responses(response_file)