from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'

  response = requests.get(url)

  return response.json()

llm = ChatOpenAI()

# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 5: Invoke
response = agent_executor.invoke({"input": "What is the current temp of gurgaon"})
print(response)

print(response['output'])




# =============================================================================
# NOTES
# =============================================================================

"""
1. Imports
----------

from langchain_openai import ChatOpenAI

Creates a LangChain wrapper around OpenAI chat models.

Example:

llm = ChatOpenAI()

After creating it, we can call

llm.invoke("Hello")

to get a response from the LLM.

We can also specify parameters like

ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

--------------------------------------------------------------------------

from langchain_core.tools import tool

tool is a decorator.

Decorators in Python are written using the @ symbol.

Example:

@tool
def add(a, b):
    return a + b

The @tool decorator converts a normal Python function into a LangChain Tool.

Without @tool:

The function is just Python code.

With @tool:

The LLM can discover it, understand its purpose, and decide when to call it.

--------------------------------------------------------------------------

import requests

requests is one of Python's most popular HTTP libraries.

It is used for communicating with APIs over the internet.

Common methods:

requests.get(url) - Retrieves data.
requests.post(url) - Sends data.
requests.put(url) - Updates data.
requests.delete(url) - Deletes data.

Example:

response = requests.get("https://example.com")

--------------------------------------------------------------------------

from langchain_community.tools import DuckDuckGoSearchRun

This is a pre-built search tool.

Instead of creating our own search API, LangChain provides ready-made tools.

Calling
search_tool.run("Latest AI news") returns search results from DuckDuckGo.
The LLM can decide when to use this tool automatically.

--------------------------------------------------------------------------

from langchain.agents import create_react_agent, AgentExecutor

These are the two main components of this example.

create_react_agent(...) - Builds the reasoning agent.
AgentExecutor(...) - Executes the agent and manages its interaction with tools.

Think of them like this:

Agent
↓
Decides WHAT to do.

Executor
↓
Actually runs the agent and its tools.

--------------------------------------------------------------------------

from langchain import hub

LangChain Hub is an online repository of reusable prompts.

Instead of writing prompts manually, we can pull well-tested prompts.

Example:

prompt = hub.pull("hwchase17/react")

This downloads the standard ReAct prompt.

--------------------------------------------------------------------------

from dotenv import load_dotenv

Loads API keys and other environment variables from the .env file.

Example:

OPENAI_API_KEY=xxxxxxxx

This keeps secrets outside the source code.

--------------------------------------------------------------------------
2. Tool Creation
----------------

search_tool = DuckDuckGoSearchRun()

Creates an instance of the DuckDuckGo search tool.

This object can now be passed to an agent.

Example:

tools = [search_tool]

The LLM can decide when to search the web.

--------------------------------------------------------------------------
3. @tool Decorator
-----------------

@tool
def get_weather_data(city: str):

This converts the function into a LangChain Tool.

The agent now knows:

- Tool name
- Function arguments
- Description
- Return value

The function signature

city: str

tells the LLM that this tool expects one string input.

--------------------------------------------------------------------------
4. Docstrings
-------------

Inside the tool we have

"""
This function fetches the current weather data for a given city
"""

This is called a docstring.

Normally, docstrings explain code to programmers.

When using @tool, LangChain also exposes the docstring to the LLM.

The model uses it to understand:

- What the tool does
- When it should use it

Good docstrings greatly improve tool selection.

--------------------------------------------------------------------------
5. f-Strings
------------

url = f"...{city}"

The "f" before a string means formatted string.

Example:

city = "Delhi"

f"Hello {city}"

becomes

"Hello Delhi"

This is the standard way of inserting variables into strings.

--------------------------------------------------------------------------
6. Weather API Request
----------------------

response = requests.get(url)

Makes an HTTP GET request to the WeatherStack API.

The flow is

Python
   ↓
WeatherStack Server
   ↓
JSON Response

The server sends back weather information.

--------------------------------------------------------------------------
7. JSON
-------

return response.json()

Most APIs return JSON.

JSON stands for

JavaScript Object Notation

It looks like

{
    "temperature": 34,
    "humidity": 60
}

response.json()

converts that JSON into a Python dictionary.

Instead of returning raw text, we return structured data.

--------------------------------------------------------------------------
8. Creating the LLM
-------------------

llm = ChatOpenAI()

Creates the language model object.

This object will perform reasoning and decide when to call tools.

--------------------------------------------------------------------------
9. LangChain Hub
----------------

prompt = hub.pull("hwchase17/react")

Downloads the official ReAct prompt.

Advantages:

- Already tested
- Maintained by LangChain
- No need to write complex prompts manually

If the prompt changes in the future, pulling it again gets the latest version.

--------------------------------------------------------------------------
10. What is ReAct?
-----------------

ReAct stands for

Reason + Act

Instead of answering immediately, the model repeatedly follows this loop:

Question

↓

Thought
"I should search."

↓

Action
Call Search Tool

↓

Observation
Receives search results

↓

Thought
"I now have enough information."

↓

Final Answer

The model alternates between reasoning and tool usage until it reaches a final answer.

--------------------------------------------------------------------------
11. Creating the Agent
----------------------

agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

This creates the reasoning agent.

It combines:

- The LLM
- Available tools
- Prompt

The agent itself does not execute tools.

It only decides:

"What should happen next?"

--------------------------------------------------------------------------
12. Available Tools
-------------------

tools=[search_tool, get_weather_data]

The list defines every tool the agent is allowed to use.

If a tool is not included here, the agent cannot call it.

For example,

Question:
"What is today's weather?"

The agent may choose

get_weather_data()

Question:
"When was Dhadak 2 released?"

The agent may choose

DuckDuckGo Search

--------------------------------------------------------------------------
13. AgentExecutor
----------------

agent_executor = AgentExecutor(...)

The executor manages the complete execution loop.

Responsibilities:

- Runs the agent.
- Executes selected tools.
- Sends tool outputs back to the LLM.
- Repeats until the agent reaches a final answer.

Think of it as the "runtime engine" for the agent.

--------------------------------------------------------------------------
14. verbose=True
----------------

verbose=True

Prints every reasoning step.

You will see output similar to

Thought:
I should search.

Action:
DuckDuckGo Search

Observation:
Movie released on...

Thought:
I now know the answer.

Final Answer:
...

This is extremely useful for debugging and learning.

--------------------------------------------------------------------------
15. max_iterations
------------------

max_iterations=5

Limits how many reasoning/tool loops the agent may perform.

Without this limit, a poorly behaving agent could get stuck in an infinite loop.

Example:

Iteration 1
Search

Iteration 2
Weather Tool

Iteration 3
Search Again

...

After five iterations, execution stops.

--------------------------------------------------------------------------
16. invoke()
------------

response = agent_executor.invoke(
    {
        "input": "What is the current temp of gurgaon"
    }
)

invoke() starts the entire agent.

Input is provided as a dictionary.

The key must be

"input"

because that is what the ReAct prompt expects.

--------------------------------------------------------------------------
17. Agent Execution Flow
------------------------

Suppose the question is

"What is the current temp of gurgaon?"

The execution looks like

User Question
      |
      v
ReAct Agent
      |
      v
Thought:
"I need weather data."
      |
      v
Calls get_weather_data()
      |
      v
Weather API
      |
      v
Returns JSON
      |
      v
LLM Reads JSON
      |
      v
Produces Final Answer

--------------------------------------------------------------------------
18. Another Example
-------------------

Question:

"Identify the birthplace city of Kalpana Chawla and give its current temperature."

Execution

User Question
      |
      v
LLM

Thought:
"I don't know her birthplace."

↓

DuckDuckGo Search

↓

Observation:
"Karnal"

↓

Thought:
"I now know the city."

↓

Weather Tool

↓

Observation:
Temperature

↓

Final Answer

Notice how the agent chains multiple tools together automatically.

--------------------------------------------------------------------------
19. Agent Response
------------------

response is a dictionary.

Typical fields include

response["input"] - Original question.
response["output"] - Final answer produced by the agent.

There may also be intermediate information depending on the executor settings.

--------------------------------------------------------------------------
20. Overall Architecture
------------------------

                 User Question
                       |
                       v
              ReAct Agent (LLM)
                       |
             Chooses next action
                       |
         -----------------------------
         |                           |
         |                           |
DuckDuckGo Search          Weather Tool
         |                           |
         -----------Results-----------
                     |
                     v
                 ReAct Agent
                     |
          More reasoning if needed
                     |
                     v
               Final Answer

The important idea is that the LLM does not directly know current information
like today's weather or recent movie release dates.

Instead, it reasons about what information it needs, chooses the appropriate
tool, observes the tool's output, and then uses that information to generate
the final answer.
"""