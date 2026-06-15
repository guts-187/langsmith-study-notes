from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatOpenAI()
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)



# =============================================================================
# NOTES
# =============================================================================

"""


14. Key Takeaways
----------------

• PromptTemplate creates reusable prompts using placeholders.
• ChatOpenAI is the interface for interacting with OpenAI's chat models.
• StrOutputParser converts an AIMessage into a plain Python string.
• LCEL uses the pipe (|) operator to connect components into a pipeline.
• invoke() executes a Runnable (or an entire chain).
• Every component has one responsibility:
    - PromptTemplate → Format the prompt
    - ChatOpenAI → Generate a response
    - StrOutputParser → Extract the text

This "Prompt → Model → Parser" pattern is the foundation of many LangChain
applications and is the simplest example of building workflows using LCEL.


1. Overview
-----------

This is one of the simplest possible LangChain applications.

The flow is:

User Input
    |
    v
PromptTemplate
    |
    v
ChatOpenAI
    |
    v
StrOutputParser
    |
    v
Final Python String

Instead of manually calling the LLM and extracting the response, LangChain
allows us to build a reusable pipeline (called a chain) where each component
performs one specific task.

--------------------------------------------------------------------------

2. load_dotenv()
----------------

load_dotenv()

Loads environment variables from a .env file.

Example:

OPENAI_API_KEY=xxxxxxxxxxxxxxxx

This allows ChatOpenAI to automatically find the API key without hardcoding
it into the source code.

Benefits:
- Better security
- Easier configuration
- API keys are not committed to GitHub

--------------------------------------------------------------------------

3. PromptTemplate
-----------------

PromptTemplate is used to create reusable prompts.

Instead of writing
question = "What is the capital of Peru?"

we define a template
PromptTemplate.from_template("{question}")

The placeholder {question} is called a template variable.
When the chain runs, {"question": "What is the capital of Peru?"} gets substituted automatically.

The final prompt becomes "What is the capital of Peru?". This makes prompts reusable for different inputs.

--------------------------------------------------------------------------

4. Why use PromptTemplate?
--------------------------

Without PromptTemplate:

prompt = f"What is the capital of Peru?"

Every time we want a new question, we have to manually create another string.

With PromptTemplate:

prompt = PromptTemplate.from_template("{question}")

Now we can reuse the same template for any question.

Example:

chain.invoke({"question": "Who invented Python?"})

↓

Prompt becomes

"Who invented Python?"

Example:

chain.invoke({"question": "Explain Neural Networks."})

↓

Prompt becomes

"Explain Neural Networks."

One template works for unlimited inputs.

--------------------------------------------------------------------------

5. ChatOpenAI
-------------

model = ChatOpenAI() - Creates a LangChain wrapper around an OpenAI chat model.

Internally, it communicates with the OpenAI API.

Common parameters include:

ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

model
- Specifies which LLM to use.

temperature
- Controls randomness.

Lower temperature (0)
- More deterministic
- Better for factual tasks

Higher temperature (1 or above)
- More creative
- Better for storytelling and brainstorming

--------------------------------------------------------------------------

6. StrOutputParser
------------------

LLMs return structured response objects.
For example, model.invoke("Hello") returns an AIMessage object.

That object contains several fields like
- content
- metadata
- token usage

Most of the time, we only want the generated text.

StrOutputParser extracts only

response.content

and returns it as a normal Python string.

Without it:

AIMessage(
    content="Lima"
)

With it:

"Lima"

This makes the output much easier to use.

--------------------------------------------------------------------------

7. LCEL (LangChain Expression Language)
---------------------------------------

This line

chain = prompt | model | parser

uses LCEL.

LCEL allows us to connect components together using the

|

(pipe) operator.

Think of it like a pipeline.

Output from one component automatically becomes the input of the next.

General pattern:

Component A
      |
      v
Component B
      |
      v
Component C

The pipe operator does NOT mean "or" here.

It means

"Send the output of the left component into the right component."

--------------------------------------------------------------------------

8. Understanding the Pipe Operator
----------------------------------

This chain

prompt | model | parser

is equivalent to

Step 1

formatted_prompt = prompt.invoke(input)

↓

Step 2

response = model.invoke(formatted_prompt)

↓

Step 3

final_answer = parser.invoke(response)

LCEL simply combines all of these steps into one readable expression.

This is similar to Unix pipes:

Command A

↓

Command B

↓

Command C

--------------------------------------------------------------------------

9. invoke()
-----------

Every Runnable in LangChain implements

.invoke()

It executes that component.

Example:

prompt.invoke({"question": "Hi"})

↓

Returns the formatted prompt.

model.invoke(prompt)

↓

Returns an AIMessage.

parser.invoke(message)

↓

Returns a string.

Since the entire pipeline is also a Runnable,

chain.invoke(...)

runs every component in order.

--------------------------------------------------------------------------

10. Why is the Input a Dictionary?
----------------------------------

Notice

chain.invoke(
    {
        "question": "What is the capital of Peru?"
    }
)

PromptTemplate expects variables.

Our template contains

{question}

Therefore LangChain needs a dictionary mapping

Variable Name

↓

Value

Example:

{
    "question": "..."
}

If the template had

"{city}"

then we'd invoke

{
    "city": "Delhi"
}

If multiple variables existed

"{city} is in {country}"

then we'd provide

{
    "city": "Delhi",
    "country": "India"
}

--------------------------------------------------------------------------

11. End-to-End Execution
------------------------

Execution proceeds as follows:

Input Dictionary

{
    "question":
    "What is the capital of Peru?"
}

↓

PromptTemplate

Produces

"What is the capital of Peru?"

↓

ChatOpenAI

Sends the prompt to the OpenAI API.

↓

Returns an AIMessage object.

↓

StrOutputParser

Extracts only the text.

↓

Final Result

"Lima"

--------------------------------------------------------------------------

12. Overall Flow
----------------

             User Input
                  |
                  v
      {"question": "..."}
                  |
                  v
         PromptTemplate
                  |
      Formats the prompt
                  |
                  v
          ChatOpenAI LLM
                  |
      AIMessage Response
                  |
                  v
       StrOutputParser
                  |
                  v
        Python String Output

--------------------------------------------------------------------------

13. Why Build Chains?
---------------------

Even though this example is small, the same idea scales to much larger systems.

A chain can include:

Prompt

↓

Retriever (RAG)

↓

LLM

↓

Output Parser

↓

Structured Output

↓

Another LLM

↓

Database

↓

API Call

↓

Final Answer

The pipe operator makes these complex workflows much easier to read and maintain.

--------------------------------------------------------------------------

14. Key Takeaways
----------------

• PromptTemplate creates reusable prompts using placeholders.
• ChatOpenAI is the interface for interacting with OpenAI's chat models.
• StrOutputParser converts an AIMessage into a plain Python string.
• LCEL uses the pipe (|) operator to connect components into a pipeline.
• invoke() executes a Runnable (or an entire chain).
• Every component has one responsibility:
    - PromptTemplate → Format the prompt
    - ChatOpenAI → Generate a response
    - StrOutputParser → Extract the text

This "Prompt → Model → Parser" pattern is the foundation of many LangChain
applications and is the simplest example of building workflows using LCEL.
"""