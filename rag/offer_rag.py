
# Built once per process, not per call (fix: was re-embedding the whole corpus every call)
_DOCS = None
_INDEX = None


def get_index():
    global _DOCS, _INDEX
    if _INDEX is None:
        _DOCS, embeddings = embed_chunks()
        _INDEX = index_embeddings(embeddings)
    return _DOCS, _INDEX


def answer_customer_rag(customer_context: dict, question: str, include_profile: bool = True) -> dict:
    docs, index = get_index()

    # Put the tier in the retrieval query so the tier-benefits chunk ranks higher
    tier = customer_context.get("loyalty_tier", "")
    query = f"{tier} tier loyalty benefits. {question}"
    question_embedding = embed_query(query)

    # FAISS needs a 2-D float32 array: one row per query
    scores, indeces = index.search(np.array([question_embedding], dtype="float32"), 6)

    retrieved_chunks = []
    chunk_ids = []
    for idx in indeces[0]:          # row 0 = results for our single query
        if idx == -1:
            continue
        retrieved_chunks.append(docs[idx].page_content)
        chunk_ids.append(docs[idx].metadata["id"])

    llm_prompt = generate_prompt(retrieved_chunks, customer_context)

    # keyword args: question is the user message, llm_prompt is the system prompt
    answer = generate_with_claude_haiku(prompt=question, system_prompt=llm_prompt)

    contexts = list(retrieved_chunks)
    if include_profile:
        # RAGAS needs strings; the profile lets the judge verify profile facts in the offer
        contexts.append(f"Customer profile: {customer_context}")

    output = {
        "answer": answer,
        "contexts": contexts,
        "chunk_ids": chunk_ids,     # harness ignores this; needed for track-b-examples.md
    }
    return output


from langchain_text_splitters import MarkdownHeaderTextSplitter
import boto3
from botocore.exceptions import ClientError
import json
from botocore.config import Config
from pathlib import Path
import faiss
import numpy as np

MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DIMENSIONS=1024
NORMALIZE=True

def generate_with_claude_haiku(prompt: str, system_prompt: str, temperature: float = 0.2, max_tokens: int = 1000):
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"  # Or your chosen Bedrock region
    )
    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}]
        }
    ]

    system = [{"text": system_prompt}] if system_prompt else []

    # Configure inference: explicitly include temperature, NEVER topP
    inference_config = {
        "temperature": temperature,
        "maxTokens": max_tokens
    }

    try:
        response = bedrock_runtime.converse(
            modelId=LLM_MODEL_ID,
            messages=messages,
            system=system,
            inferenceConfig=inference_config
        )

        # Extract text output from the Converse API response structure
        output_text = response["output"]["message"]["content"][0]["text"]
        
        # Optional: inspect token consumption
        usage = response.get("usage", {})
        print(f"[Usage] Input tokens: {usage.get('inputTokens')}, Output tokens: {usage.get('outputTokens')}")

        return output_text

    except ClientError as e:
        print(f"AWS Bedrock error: {e.response['Error']['Message']}")
        raise

def generate_prompt(retrieved_chunks,customer_context):
    prompt = f"""
                # Role
                Your job is to answer customer questions using retrieved context via a RAG System.

                # Customer Context
                {customer_context}

                # Context
                Using RAG we have retrieved the most relevant information from our documents

            """

    for i in range(len(retrieved_chunks)):
        prompt += f"{i}: {retrieved_chunks[i]}\n"

    prompt += "# Rules\n- Response must be 2-3 sentences long, no more, no less\n- Use only benefits this tier has\n- recommend one specific product category from the customer's top_categories\n- do not use greetings or madeup deadlines"

    return prompt

def create_embeddings(docs):
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",  # Adjust to your AWS region
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"})
    )
    embeddings = []
    for doc in docs:
        text = doc.page_content

        payload = {
            "inputText": text,
            "dimensions": DIMENSIONS,
            "normalize": NORMALIZE
        }

        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )

        response_body = json.loads(response["body"].read().decode("utf-8"))
        embeddings.append(response_body["embedding"])
    return embeddings

def embed_query(query):
    bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",  # Adjust to your AWS region
            config=Config(retries={"max_attempts": 5, "mode": "adaptive"})
        )
    payload = {
            "inputText": query,
            "dimensions": DIMENSIONS,
            "normalize": NORMALIZE
        }

    response = bedrock_runtime.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )

    response_body = json.loads(response["body"].read().decode("utf-8"))
    embedding = (response_body["embedding"])

    return embedding

    
def embed_chunks():
    folder_path = Path(__file__).resolve().parent.parent / "northstar-policy-docs"

    file_contents = []

    for file_path in folder_path.glob("*.md"):
        try:
            content = file_path.read_text(encoding="utf-8")
            print(f"File: {file_path.name} ({len(content)} characters)")
            file_contents.append((file_path.stem, content))
            # Process 'content' here
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    all_chunk_docs = []
    for source, md_file in file_contents:
        chunk_docs = chunk_markdown(md_file, source)
        all_chunk_docs.extend(chunk_docs)

    embeddings = create_embeddings(all_chunk_docs)

    return all_chunk_docs,embeddings

def index_embeddings(embeddings):
    # Plain flat index: search returns row positions, so docs[i] is the chunk.
    # (add_with_ids needs int64 IDs; our IDs are strings, so we keep them in metadata.)
    index = faiss.IndexFlatIP(DIMENSIONS)
    index.add(np.array(embeddings, dtype="float32"))
    print(f"Total vectors indexed: {index.ntotal}")
    return index





def chunk_markdown(md_text, source):


    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # Keep headers inside content for context
    )

    docs = markdown_splitter.split_text(md_text)

    # Assign deterministic hash or slug IDs to each Document chunk
    for i, doc in enumerate(docs):
        # Readable ID: source file + chunk position, e.g. "return-policy#3"
        doc.metadata["source"] = source
        doc.metadata["id"] = f"{source}#{i}"

    return docs
    # `docs` is a standard Python list of LangChain `Document` objects:
    #
    #   List[langchain_core.documents.Document]
    #
    # Each `Document` in the list has two primary attributes:
    #
    # 1. doc.page_content (str):
    #    The raw Markdown text under that section (including or excluding the 
    #    header based on the `strip_headers` setting).
    #
    # 2. doc.metadata (dict):
    #    A dictionary preserving the hierarchy of headers leading up to that chunk,
    #    along with the custom 'id' generated in the loop.
    #
    # Example representation of `docs` after running the code:
    # [
    #     Document(
    #         page_content="# Getting Started\nHere is some introductory text.",
    #         metadata={
    #             "Header 1": "Getting Started",
    #             "id": "e3b0c44298fc"
    #         }
    #     ),
