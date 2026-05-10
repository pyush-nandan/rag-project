import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document


Judgement_splitter = RecursiveCharacterTextSplitter(
    separators = [
        "\n\n",
        "\nHELD", "\nOrder", "\nJudgement"
        "\nFACTS", "\nIssues", "\nRatio",
        "\n", " "
    ],
    chunk_size = 600,
    chunk_overlap = 80
)

def _clean_text(text : str) -> str:
    # Rejoin hyphenated line breaks (e.g. "liabil-\nity" → "liability")
    text = re.sub(r"-\n(\S)", r"\1", text)

    # Rejoin lines that are broken mid-word without hyphen
    # Only join if the line doesn't end with sentence-ending punctuation
    text = re.sub(r"(?<![.!?:,])\n(?=[a-z])", " ", text)

    # Collapse 3+ newlines into a paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace per line
    text = "\n".join(line.strip() for line in text.splitlines())

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def ingest_data_and_get_vectorstore() -> Chroma:

    embedding_model = SentenceTransformerEmbeddings(
        model_name = "all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        collection_name = "embeddings",
        embedding_function = embedding_model,
        persist_directory = "./chroma_db"
    )

    for file_name in os.listdir("data"):
        file_path = os.path.join("data", file_name)
        existing = vectorstore._collection.get(where={"source": file_path})
        if existing["ids"]:
            # print(f"{file_name} already ingested. Skipping!")
            continue
        else:
            pages = PyPDFLoader(file_path).load()
            for page in pages:
                page.page_content = _clean_text(page.page_content)
            print(f"Loaded {len(pages)} pages from {pages}")

            if file_name.startswith("Section_"):
                chunks = [
                    Document(
                        page_content = "\n".join(p.page_content for p in pages),
                        metadata = {"source" : file_path, "filename" : file_name, "doc_type" : "statute"}
                    )
                ]
            else:
                chunks = Judgement_splitter.split_documents(pages)
                for chunk in chunks:
                    chunk.metadata["filename"] = file_name
                    chunk.metadata["doc_type"] = "judgement"
            
            print(f"Split into {len(chunks)} chunks")

            vectorstore.add_documents(chunks)
            print("Documents added to the vectorstore")   
            
    return vectorstore

