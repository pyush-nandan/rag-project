from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import yaml
from retriever import get_retriever

load_dotenv()

def build_rag_chain():
    llm = ChatGroq(
        model_name = "llama-3.1-8b-instant",
        temperature = 0.2,
        max_tokens = 1000,
        timeout = None,
        max_retries = 3
    )

    with open("prompts.yaml", "r") as f:
        prompt_dict = yaml.safe_load(f)

    active_version = prompt_dict["active_version"]

    prompt = ChatPromptTemplate.from_messages(
        [("system", prompt_dict["prompts"][active_version]["system"]),
        ("human", prompt_dict["prompts"][active_version]["human"])]
    )

    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(get_retriever(), doc_chain)
    return rag_chain

def run_rag():
    rag_chain = build_rag_chain()
    flag = True
    while flag:
        query = input("\n\nAsk your question: ")
        response = rag_chain.invoke({"input": query})
        for i, doc in enumerate(response["context"]):
            print(f"Chunk {i+1}: {doc.page_content}")
            print("---")
        print("\n\n\n -----------Answer----------- \n\n\n")
        print(response["answer"])

        more_questions = input("\n\nDo you want to ask more questions (y/n): ")
        if more_questions.lower() == "n":
            flag = False

if __name__ == "__main__":
    run_rag()