# pip install -U langgraph langchain-openai pydantic python-dotenv langsmith

import operator
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langsmith import traceable
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# ---------- Setup ----------
load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------- Structured schema & model ----------
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

# ---------- Sample essay ----------
essay2 = """India and AI Time

Now world change very fast because new tech call Artificial Intel… something (AI). India also want become big in this AI thing. If work hard, India can go top. But if no careful, India go back.

India have many good. We have smart student, many engine-ear, and good IT peoples. Big company like TCS, Infosys, Wipro already use AI. Government also do program “AI for All”. It want AI in farm, doctor place, school and transport.

In farm, AI help farmer know when to put seed, when rain come, how stop bug. In health, AI help doctor see sick early. In school, AI help student learn good. Government office use AI to find bad people and work fast.

But problem come also. First is many villager no have phone or internet. So AI not help them. Second, many people lose job because AI and machine do work. Poor people get more bad.

One more big problem is privacy. AI need big big data. Who take care? India still make data rule. If no strong rule, AI do bad.

India must all people together – govern, school, company and normal people. We teach AI and make sure AI not bad. Also talk to other country and learn from them.

If India use AI good way, we become strong, help poor and make better life. But if only rich use AI, and poor no get, then big bad thing happen.

So, in short, AI time in India have many hope and many danger. We must go right road. AI must help all people, not only some. Then India grow big and world say "good job India".
"""

# ---------- LangGraph state ----------
class UPSCState(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add]  # merges parallel lists
    avg_score: float

# ---------- Traced node functions ----------
@traceable(name="evaluate_language_fn", tags=["dimension:language"], metadata={"dimension": "language"})
def evaluate_language(state: UPSCState):
    prompt = (
        "Evaluate the language quality of the following essay and provide feedback "
        "and assign a score out of 10.\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"language_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="evaluate_analysis_fn", tags=["dimension:analysis"], metadata={"dimension": "analysis"})
def evaluate_analysis(state: UPSCState):
    prompt = (
        "Evaluate the depth of analysis of the following essay and provide feedback "
        "and assign a score out of 10.\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"analysis_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="evaluate_thought_fn", tags=["dimension:clarity"], metadata={"dimension": "clarity_of_thought"})
def evaluate_thought(state: UPSCState):
    prompt = (
        "Evaluate the clarity of thought of the following essay and provide feedback "
        "and assign a score out of 10.\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"clarity_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="final_evaluation_fn", tags=["aggregate"])
def final_evaluation(state: UPSCState):
    prompt = (
        "Based on the following feedback, create a summarized overall feedback.\n\n"
        f"Language feedback: {state.get('language_feedback','')}\n"
        f"Depth of analysis feedback: {state.get('analysis_feedback','')}\n"
        f"Clarity of thought feedback: {state.get('clarity_feedback','')}\n"
    )
    overall = model.invoke(prompt).content
    scores = state.get("individual_scores", []) or []
    avg = (sum(scores) / len(scores)) if scores else 0.0
    return {"overall_feedback": overall, "avg_score": avg}

# ---------- Build graph ----------
graph = StateGraph(UPSCState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# Fan-out → join
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("final_evaluation", END)

workflow = graph.compile()

# ---------- Direct invoke without wrapper ----------
if __name__ == "__main__":
    result = workflow.invoke(
        {"essay": essay2},
        config={
            "run_name": "evaluate_upsc_essay",  # becomes root run name
            "tags": ["essay", "langgraph", "evaluation"],
            "metadata": {
                "essay_length": len(essay2),
                "model": "gpt-4o-mini",
                "dimensions": ["language", "analysis", "clarity"],
            },
        },
    )

    print("\n=== Evaluation Results ===")
    print("Language feedback:\n", result.get("language_feedback", ""), "\n")
    print("Analysis feedback:\n", result.get("analysis_feedback", ""), "\n")
    print("Clarity feedback:\n", result.get("clarity_feedback", ""), "\n")
    print("Overall feedback:\n", result.get("overall_feedback", ""), "\n")
    print("Individual scores:", result.get("individual_scores", []))
    print("Average score:", result.get("avg_score", 0.0))



# =============================================================================
# NOTES
# =============================================================================

"""
1. Imports
----------

import operator
- Python's built-in module containing functions corresponding to operators.
- Instead of using '+' directly, we can use operator.add.
- Common functions:
    operator.add(a, b)      -> a + b
    operator.sub(a, b)      -> a - b
    operator.mul(a, b)      -> a * b
    operator.truediv(a, b)  -> a / b
    operator.eq(a, b)       -> a == b

- In LangGraph, operator.add is commonly used for merging values coming from
  parallel nodes. If two nodes return:
      [8]
      [9]
  then operator.add combines them into:
      [8, 9]

--------------------------------------------------------------------------

from typing import TypedDict, Annotated, List

These are Python type hints.

List[int]
- Means a list containing integers.
- Example:
    scores: List[int] = [8, 9, 10]

TypedDict
- Defines the structure (schema) of a dictionary.
- Similar to defining fields in a class, except the object is still a dictionary.

Example:
class Student(TypedDict):
    name: str
    age: int

student = {
    "name": "Rahul",
    "age": 21
}

This helps IDEs, static type checkers, and LangGraph know what keys exist.

Annotated
- Adds extra metadata to a type.
- General syntax:

    Annotated[actual_type, extra_information]

Normally:

    List[int]

becomes

    Annotated[List[int], operator.add]

The actual type is still List[int], but the extra metadata tells LangGraph
how this field should behave.

Here,

    individual_scores: Annotated[List[int], operator.add]

means

"This is a list of integers. If multiple parallel nodes return this field,
merge them using operator.add."

Without Annotated, LangGraph would not know how to merge values coming from
parallel execution.

--------------------------------------------------------------------------

2. dotenv
----------

load_dotenv()

Loads variables from a .env file into environment variables.

Example .env

OPENAI_API_KEY=xxxxxxxx
LANGSMITH_API_KEY=xxxxxxxx

Instead of hardcoding secrets inside the code, we keep them in .env files.

Advantages:
- More secure
- Easy to switch keys
- Never commit API keys to GitHub

--------------------------------------------------------------------------

3. Pydantic
-----------

Pydantic is a data validation library.

class EvaluationSchema(BaseModel):

BaseModel
- Base class for defining structured data.

Field(...)
- Adds validation and descriptions.

Example:

score: int = Field(
    description="Score out of 10",
    ge=0,
    le=10
)

ge = greater than or equal to
le = less than or equal to

So score must satisfy:

0 <= score <= 10

If the LLM produces 15 or -2, validation fails.

Descriptions are also sent to the LLM when using structured output, helping it
understand exactly what each field represents.

--------------------------------------------------------------------------

4. Structured Output
--------------------

structured_model = model.with_structured_output(EvaluationSchema)

Normally:

response = model.invoke(prompt)

returns text.

With structured output,

response = structured_model.invoke(prompt)

returns an EvaluationSchema object.

Instead of parsing text manually, we can directly access

response.feedback
response.score

Benefits:
- Reliable
- No regex
- No string parsing
- Validated output

--------------------------------------------------------------------------

5. TypedDict State
------------------

class UPSCState(TypedDict, total=False):

This defines the shared state flowing through the graph.

Every node receives

state

and returns

{
    ...
}

The returned values are merged into this shared state.

total=False means every field is optional.

Without it, every key would be required.

--------------------------------------------------------------------------

6. LangSmith @traceable
-----------------------

@traceable(...)

Automatically records function execution inside LangSmith.

It tracks:
- Inputs
- Outputs
- Runtime
- Errors
- Metadata
- Parent-child relationships

Example:

@traceable(
    name="evaluate_language_fn",
    tags=["dimension:language"],
    metadata={"dimension": "language"}
)

name
- Name shown in LangSmith.

tags
- Used for filtering runs.

metadata
- Stores additional searchable information.

--------------------------------------------------------------------------

7. Prompt Construction
----------------------

Each evaluation function creates a prompt like:

"Evaluate the language quality..."

+ essay

This tells the LLM exactly what aspect to evaluate.

Different prompts produce different evaluations while using the same model.

--------------------------------------------------------------------------

8. Returning Dictionaries
--------------------------

Every node returns a dictionary.

Example:

return {
    "language_feedback": out.feedback,
    "individual_scores": [out.score]
}

LangGraph merges this dictionary into the graph state.

Only the returned keys are updated.

--------------------------------------------------------------------------

9. Why score is inside a list?
------------------------------

Notice:

"individual_scores": [out.score]

instead of

"individual_scores": out.score

Reason:

Three nodes execute in parallel.

Each returns

[8]

[9]

[7]

Since the state uses

Annotated[List[int], operator.add]

LangGraph merges them into

[8, 9, 7]

If integers were returned directly, LangGraph wouldn't know how to combine them.

--------------------------------------------------------------------------

10. Graph Construction
----------------------

graph = StateGraph(UPSCState)
Creates a graph whose shared state is UPSCState.

graph.add_node(name, function)

Registers functions as nodes.

Example:

graph.add_node(
    "evaluate_language",
    evaluate_language
)

Now the graph knows this node exists.

--------------------------------------------------------------------------

11. START and END
-----------------

START - Entry point of the graph.
END - Graph finishes here.

Think of them as virtual nodes.

START
   |
 node1
   |
 END

--------------------------------------------------------------------------

12. Edges
---------

graph.add_edge(A, B)

Creates a connection.

Execution flows

A -> B

Example:

START
   |
Language

means

Language begins immediately after START.

--------------------------------------------------------------------------

13. Parallel Execution (Fan-Out)
--------------------------------

These three edges

START -> Language
START -> Analysis
START -> Thought

create parallel execution.

Instead of

Language
then
Analysis
then
Thought

all three execute simultaneously.

This reduces overall execution time.

--------------------------------------------------------------------------

14. Join
--------

All three nodes connect to

Final Evaluation

Language ----\
Analysis ----- > Final Evaluation
Thought ------/

LangGraph automatically waits until all incoming nodes finish before executing
Final Evaluation.

--------------------------------------------------------------------------

15. compile()
-------------

workflow = graph.compile()
workflow.invoke(...)

Converts the graph definition into an executable workflow.

Before compile(): - only blueprint
After compile(): - executable graph

--------------------------------------------------------------------------

16. invoke()
------------

workflow.invoke(...)
Runs the graph.

Input:

{
    "essay": essay2
}

becomes the initial graph state. Every node receives this state.

--------------------------------------------------------------------------

17. config
----------

config={...}

Provides execution metadata.

run_name - Root run name in LangSmith.
tags - Helps organize runs.
metadata - Stores additional information.

Example:

essay_length
model
dimensions

These do not affect model output. They are useful for monitoring and searching in LangSmith.

--------------------------------------------------------------------------

18. .get() instead of []
----------

Instead of state["language_feedback"] the code often uses state.get("language_feedback", "")

Difference:

state["x"] - Raises KeyError if missing.
state.get("x", default) - Returns default if missing.

Safer for optional fields.

--------------------------------------------------------------------------

19. Overall Flow
----------------

essay
   |
   v

START

   |
   |----------------------------|
   |                            |
Language                    Analysis
   |                            |
   |                            |
Thought ------------------------|
             |
             v
      Final Evaluation
             |
             v
            END

The three evaluation nodes run in parallel.

Each produces:
- Feedback
- One score

The scores are merged into one list using operator.add.

The final node:
- Summarizes all feedback.
- Computes the average score.
- Returns the final state.
"""