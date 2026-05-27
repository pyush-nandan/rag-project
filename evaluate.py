import os
import sys
import pandas as pd
import json
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt
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


@retry(wait=wait_exponential(multiplier=1, min=4, max=15), stop=stop_after_attempt(5))
async def safe_ainvoke(rag_chain, query):
    return await rag_chain.ainvoke({"input": query})

async def generate_responses_async(testset, rag_chain):
    CACHE_FILE = "rag_responses_cache.json"
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    eval_list = []
    total = len(testset)
    batch_size = 2  # Process 2 at a time

    for i in range(0, total, batch_size):
        batch = testset.iloc[i:i+batch_size]
        tasks = []
        rows_to_process = []
        
        for row in batch.itertuples(index=False):
            if row.question in cache:
                print(f"  [Cached] {row.question[:80]}...")
                eval_list.append(cache[row.question])
            else:
                print(f"  [API] {row.question[:80]}...")
                tasks.append(safe_ainvoke(rag_chain, row.question))
                rows_to_process.append(row)
        
        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for row, response in zip(rows_to_process, responses):
                if isinstance(response, Exception):
                    print(f"  Error on '{row.question[:40]}': {response}")
                else:
                    result_dict = {
                        "user_input": row.question,
                        "response": response["answer"],
                        "reference": row.ground_truth,
                        "reference_contexts": row.context.split(CONTEXT_SEPARATOR),
                        "retrieved_contexts": [doc.page_content for doc in response["context"]],
                    }
                    cache[row.question] = result_dict
                    eval_list.append(result_dict)
            
            # Save cache to disk
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f, indent=2)
            
            # Throttle to stay within 30 RPM (2 requests / 5 seconds = 24 RPM)
            print("  Sleeping for 5s to avoid rate limits...")
            await asyncio.sleep(5)
            
    return eval_list


def build_eval_dataset(testset_path: str = "testset.csv") -> EvaluationDataset:
    """Run the RAG chain on every test question and build an EvaluationDataset."""
    testset = pd.read_csv(testset_path)
    rag_chain = build_rag_chain()
    
    # Run the throttled async generation
    eval_list = asyncio.run(generate_responses_async(testset, rag_chain))
    
    print(f"\nBuilt {len(eval_list)}/{len(testset)} evaluation samples.")
    return EvaluationDataset.from_list(eval_list)


def run_evaluation():
    """Run RAGAS evaluation and print + save the results."""
    print("=== Building evaluation dataset ===")
    eval_dataset = build_eval_dataset()

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

    overall_faithfulness = result_df["faithfulness"].mean()
    if overall_faithfulness < FAITHFULNESS_THRESHOLD:
        print(f"Quality Gate Failed: faithfulness {overall_faithfulness:.4f} < {FAITHFULNESS_THRESHOLD}")
        sys.exit(1)
    else:
        print(f"Quality Gate Passed: faithfulness {overall_faithfulness:.4f}")
        sys.exit(0)
    
    #quality gate

if __name__ == "__main__":
    run_evaluation()