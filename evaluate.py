import os
import sys
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.run_config import RunConfig
from main import build_rag_chain

load_dotenv()

CONTEXT_SEPARATOR = " + "
FAITHFULNESS_THRESHOLD = 0.75


def build_eval_dataset(testset_path: str = "testset.csv") -> EvaluationDataset:
    """Run the RAG chain on every test question and build an EvaluationDataset."""
    testset = pd.read_csv(testset_path)
    rag_chain = build_rag_chain()

    eval_list = []
    total = len(testset)

    for row in testset.itertuples(index=False):
        idx = eval_list.__len__()
        print(f"  [{idx + 1}/{total}] {row.question[:80]}...")
        try:
            response = rag_chain.invoke({"input": row.question})
            eval_list.append({
                "user_input": row.question,
                "response": response["answer"],
                "reference": row.ground_truth,
                "reference_contexts": row.context.split(CONTEXT_SEPARATOR),
                "retrieved_contexts": [doc.page_content for doc in response["context"]],
            })
        except Exception as e:
            print(f"    ⚠ Error on row {idx}: {e}")

    print(f"\nBuilt {len(eval_list)}/{total} evaluation samples.")
    return EvaluationDataset.from_list(eval_list)


def run_evaluation():
    """Run RAGAS evaluation and print + save the results."""
    print("=== Building evaluation dataset ===")
    eval_dataset = build_eval_dataset()

    # --- LLM for evaluation (Groq via OpenAI-compatible endpoint) ---
    client = AsyncOpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        max_retries=5,
        timeout=120.0,
    )

    llm = llm_factory(
        model="llama-3.1-8b-instant",
        provider="openai",
        client=client,
    )

    embedding_model = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

    metrics = [
        Faithfulness(llm = llm),
        AnswerRelevancy(llm = llm, embeddings = embedding_model),
        ContextPrecision(llm = llm),
        ContextRecall(llm = llm)
    ]

    # Rate-limit-friendly config for Groq free tier
    run_config = RunConfig(
        max_workers=2,
        max_retries=10,
        max_wait=90,
        timeout=300,
    )

    print("\n=== Running RAGAS evaluation ===")
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embedding_model,
        run_config=run_config,
        raise_exceptions=False,
        batch_size=5,
    )

    print("\n=== Evaluation Results ===")
    print(result)

    # Persist results
    result_df = result.to_pandas()
    result_df.to_csv("evaluation_results.csv", index=False)
    print("\nResults saved to evaluation_results.csv")

    scores = result.scores
    if scores["faithfulness"] < FAITHFULNESS_THRESHOLD:
        print(f"Quality Gate failed: faithfulness {scores['faithfulness']:.4f} < {FAITHFULNESS_THRESHOLD}")
        sys.exit(1)
    else:
        print(f"Quality Gate passed: faithfulness {scores['faithfulness']:.4f}")
        sys.exit(0)



if __name__ == "__main__":
    run_evaluation()