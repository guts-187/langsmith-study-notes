# Comprehensive Notes: PDF RAG with LangChain, FAISS and LCEL

------------------------------------------------------------------------

# 1. High-Level Architecture

``` text
                PDF
                 |
          PyPDFLoader
                 |
        List[Document]
                 |
 RecursiveCharacterTextSplitter
                 |
      Small overlapping chunks
                 |
        OpenAI Embeddings
                 |
      Dense Vector Embeddings
                 |
            FAISS Index
                 |
         Similarity Search
                 |
          Retrieved Chunks
                 |
      Prompt + User Question
                 |
           ChatOpenAI LLM
                 |
            Final Answer
```

This is Retrieval-Augmented Generation (RAG).

Instead of relying only on the model's memory, we first retrieve
relevant information from a knowledge base and provide it as context.

------------------------------------------------------------------------

# 2. Imports

## os

Provides operating system utilities.

Common uses:

-   environment variables
-   file paths
-   creating directories
-   absolute paths

Example:

``` python
os.environ["OPENAI_API_KEY"]
os.path.abspath("file.pdf")
```

------------------------------------------------------------------------

## json

Converts Python objects to JSON and vice versa.

Common functions:

``` python
json.dumps(obj)
json.loads(text)
```

Used here for metadata files and cache keys.

------------------------------------------------------------------------

## hashlib

Used for hashing. A hash is a fixed-size fingerprint of data.

Common algorithms:

-   md5
-   sha1
-   sha256

This project uses SHA256 because collisions are extremely unlikely. Changing even one byte changes the hash completely.

Purpose:

-   detect file changes
-   generate cache keys

------------------------------------------------------------------------

## pathlib.Path

Modern object-oriented way of handling paths.

Instead of

``` python
os.path.join(...)
```

you can write

``` python
Path("folder") / "file.txt"
```

Cleaner and cross-platform.

------------------------------------------------------------------------

## dotenv

``` python
load_dotenv()
```

Loads API keys from the .env file.

------------------------------------------------------------------------

# 3. LangSmith Traceable

Every function marked with

``` python
@traceable
```

becomes visible inside LangSmith.

Benefits:

-   timing
-   inputs
-   outputs
-   nested execution tree
-   debugging

Notice how larger functions call traced helper functions. This creates parent-child traces.

------------------------------------------------------------------------

# 4. PyPDFLoader

``` python
PyPDFLoader(path).load()
```

Reads a PDF.

Returns

``` python
list[Document]
```

A Document contains

-   page_content
-   metadata

Example

``` python
doc.page_content
doc.metadata
```

The loader preserves page boundaries.

------------------------------------------------------------------------

# 5. Why Documents instead of Strings?

A Document stores

-   text
-   metadata

Metadata may include

-   page number
-   source
-   filename

Later you can cite pages.

------------------------------------------------------------------------

# 6. RecursiveCharacterTextSplitter

LLMs have context limits.

Large PDFs cannot be embedded or retrieved as one giant block.

Instead

100-page PDF

↓

many chunks

Example

    Chunk 1
    Chunk 2
    Chunk 3
    ...

------------------------------------------------------------------------

## chunk_size

Maximum size of every chunk.

Larger chunks

Pros

-   more context

Cons

-   noisier retrieval
-   higher embedding cost

------------------------------------------------------------------------

## chunk_overlap

Neighbouring chunks overlap.

Without overlap

    Chunk1
    -----Sentence-----
    Chunk2

The sentence could split in half.

With overlap

    Chunk1
    ------Sentence------
    Chunk2
       ------Sentence------

Information near boundaries survives.

Typical overlap

10-20%.

------------------------------------------------------------------------

# 7. Embeddings

Embedding = numerical representation of text.

Instead of storing words

    Machine Learning

the model converts them into vectors.

Example (simplified)

    [0.21, -0.55, 0.84, ...]

Texts with similar meaning produce nearby vectors.

Embeddings capture semantics instead of exact words.

------------------------------------------------------------------------

# 8. OpenAIEmbeddings

Responsible only for converting text into vectors.

It does NOT answer questions.

LLM and embedding models are separate.

Embedding pipeline

Text

↓

Embedding model

↓

Vector

↓

Stored inside FAISS

------------------------------------------------------------------------

# 9. FAISS

FAISS = Facebook AI Similarity Search.

Purpose:

Fast nearest-neighbour search over millions of vectors.

Instead of comparing against every chunk manually, FAISS performs
efficient similarity search.

------------------------------------------------------------------------

# 10. Similarity Search

User asks

"What is backpropagation?"

Question becomes embedding.

FAISS compares it with every stored vector.

Returns

Top K most similar chunks.

Usually cosine similarity or inner-product based.

------------------------------------------------------------------------

# 11. Why not send the whole PDF?

Imagine a 700-page textbook.

Problems

-   exceeds context window
-   slower
-   expensive
-   irrelevant information

Retrieval selects only relevant chunks.

------------------------------------------------------------------------

# 12. File Fingerprinting

The project computes

-   SHA256
-   size
-   modification time

Together these uniquely identify the PDF version.

If the PDF changes, the fingerprint changes, forcing index rebuild.

This prevents stale indexes.

------------------------------------------------------------------------

# 13. Cache Key

The cache depends on

-   PDF fingerprint
-   chunk size
-   overlap
-   embedding model

Changing ANY of these creates a new cache.

This is called cache invalidation.

------------------------------------------------------------------------

# 14. Why cache?

Building embeddings is expensive.

Without cache

Every run

↓

Read PDF

↓

Split

↓

Embed

↓

Build FAISS

With cache

↓

Load existing FAISS index

Startup becomes much faster.

------------------------------------------------------------------------

# 15. save_local()

Stores the FAISS index on disk.

Next execution loads it directly.

------------------------------------------------------------------------

# 16. load_local()

Loads the saved vector database.

No embedding generation required.

------------------------------------------------------------------------

# 17. Retriever

A vector store stores vectors.

A Retriever exposes a search interface.

``` python
retriever = vectorstore.as_retriever(...)
```

Now the chain simply asks

retrieve(question)

instead of manually searching.

------------------------------------------------------------------------

# 18. search_kwargs={"k":4}

k = number of chunks returned.

Small k

Pros

-   focused

Cons

-   may miss information

Large k

Pros

-   broader coverage

Cons

-   more irrelevant context

Typical values

3--8.

------------------------------------------------------------------------

# 19. ChatPromptTemplate

Instead of formatting strings manually,

LangChain lets you define reusable prompts.

Variables

    {question}
    {context}

are filled automatically.

------------------------------------------------------------------------

# 20. format_docs()

Retriever returns Document objects.

LLM expects text.

So

Documents

↓

Single string

↓

Prompt

------------------------------------------------------------------------

# 21. LCEL (LangChain Expression Language)

LCEL connects components using

``` python
|
```

The pipe operator means

output of left

↓

input of right

Example

Retriever

↓

Formatter

↓

Prompt

↓

LLM

↓

Parser

------------------------------------------------------------------------

# 22. RunnableParallel

Runs independent branches simultaneously.

Question

↓

  -----------------------------
  \| \| Retriever Original
  Question \| \| Context
  Pass-through
  -----------------------------
  ↓

  Dictionary

  This reduces latency.
  -----------------------------

# 23. RunnablePassthrough

Returns the input unchanged.

Useful when one branch needs the original input.

------------------------------------------------------------------------

# 24. RunnableLambda

Wraps a normal Python function into an LCEL Runnable.

Allows ordinary functions to participate in pipelines.

------------------------------------------------------------------------

# 25. StrOutputParser

LLM returns AIMessage objects.

Parser extracts only

    response.content

Result becomes a normal Python string.

------------------------------------------------------------------------

# 26. setup_pipeline()

Responsible for

-   load cache OR
-   build cache

It separates indexing from querying.

------------------------------------------------------------------------

# 27. setup_pipeline_and_query()

Entire RAG workflow.

Steps

1.  Load/build FAISS
2.  Create retriever
3.  Retrieve relevant chunks
4.  Format chunks
5.  Build prompt
6.  Send to LLM
7.  Parse output
8.  Return answer

------------------------------------------------------------------------

# 28. LCEL Pipeline

``` text
Question
   |
RunnablePassthrough
   |
RunnableParallel
 |               |
Retriever     Original Question
 |               |
format_docs      |
 |               |
 ---- Context + Question ----
             |
      ChatPromptTemplate
             |
        ChatOpenAI
             |
     StrOutputParser
             |
        Final String
```

------------------------------------------------------------------------

# 29. Why RAG?

Without RAG

LLM answers from training knowledge.

With RAG

LLM answers from YOUR documents.

Advantages

-   current knowledge
-   private data
-   fewer hallucinations
-   domain-specific answers

------------------------------------------------------------------------

# 30. Complete Execution Flow

``` text
PDF
 |
Load
 |
Split
 |
Embeddings
 |
FAISS
 |
Question
 |
Embedding
 |
Similarity Search
 |
Top-4 Chunks
 |
Prompt
 |
GPT-4o-mini
 |
Answer
```

This is the standard architecture used in production RAG systems. The
same pattern scales to enterprise search, chat-with-PDF, document QA,
legal assistants, financial copilots, healthcare knowledge retrieval and
many internal company knowledge bases. The main differences in
production systems are larger vector databases, reranking, hybrid search
(keyword + vector), metadata filtering, streaming responses, and
evaluation pipelines.
