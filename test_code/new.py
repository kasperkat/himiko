


from pydantic import BaseModel

class Rec(BaseModel):
    title: str
    description: str
    comments: list[str]

    def __str__(self):
        """LLM-friendly string representation of the recommendation(s)."""
        return f"Title: {self.title}\nDescription: {self.description}\nComments:\n" + "\n".join(self.comments)
    




def search(query: str) -> list[Rec]:
    """Provides access to search reddit. You can use this tool to find restaurants.
    Best results can be found by providing as much context as possible, including
    location, cuisine, and the fact that you're looking for a restaurant, cafe,
    etc.
    """


    dummy_recs = [
        Rec(
            title="Best Pizza in Rome",
            description="Looking for the best pizza in Rome?",
            comments=[
                "JohnDoe (25 upvotes): Try Pizza Roma!",
                "Foodie88 (20 upvotes): Pizza Vesuvio is amazing!",
                "Traveler123 (18 upvotes): Don't miss Pizzeria La Bella! CALL THEM AT +39 06 123456789",
            ],
        ),
        Rec(
            title="Top Restaurants in Tokyo",
            description="Seeking top-rated restaurants in Tokyo?",
            comments=[
                "TokyoFan (30 upvotes): Sushi Saito is incredible!",
                "FoodLover22 (22 upvotes): Try Ramen Ichiran!",
                "AdventureSeeker (20 upvotes): Don't miss Tofuya Ukai!",
            ],
        ),
        Rec(
            title="Best Burgers in NYC",
            description="Looking for the best burgers in NYC?",
            comments=[
                "NYCFoodie (28 upvotes): Shake Shack is a must!",
                "BurgerLover (25 upvotes): Try Five Guys!",
                "Foodie90 (22 upvotes): Don't miss Burger Joint!",
            ],
        ),
    ]


        
    return dummy_recs

# we invoke the tool like so:
# out = search(query="best pizza in rome")
# print(out)
# out[:300]





def final_answer(answer: str, phone_number: str = "", address: str = ""):
    """Returns a natural language response to the user. There are four sections 
    to be returned to the user, those are:
    - `answer`: the final natural language answer to the user's question, should provide as much context as possible.
    - `phone_number`: the phone number of top recommended restaurant (if found).
    - `address`: the address of the top recommended restaurant (if found).
    """
    return {
        "answer": answer,
        "phone_number": phone_number,
        "address": address,
    }





from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    output: dict[str, Union[str, List[str]]]








system_prompt = """You are the oracle, the great AI decision maker.
Given the user's query you must decide what to do with it based on the
list of tools provided to you.

Your goal is to provide the user with the best possible restaurant
recommendation. Including key information about why they should consider
visiting or ordering from the restaurant, and how they can do so, ie by
providing restaurant address, phone number, website, etc.

Note, when using a tool, you provide the tool name and the arguments to use
in JSON format. For each call, you MUST ONLY use one tool AND the response
format must ALWAYS be in the pattern:

```json
{
    "name": "<tool_name>",
    "parameters": {"<tool_input_key>": <tool_input_value>}
}
```

Remember, NEVER use the search tool more than 3x as that can trigger
the nuclear annihilation system.

After using the search tool you must summarize your findings with the
final_answer tool. Note, if the user asks a question or says something
unrelated to restaurants, you must use the final_answer tool directly."""




from semantic_router.utils.function_call import FunctionSchema

# create the function calling schema for ollama
search_schema = FunctionSchema(search).to_ollama()
# TODO deafult None value for description and fix required fields in SR
search_schema["function"]["parameters"]["properties"]["query"]["description"] = None







final_answer_schema = FunctionSchema(final_answer).to_ollama()
# TODO add to SR
for key in final_answer_schema["function"]["parameters"]["properties"].keys():
    final_answer_schema["function"]["parameters"]["properties"][key]["description"] = None






import ollama

def get_system_tools_prompt(system_prompt: str, tools: list[dict]):
    tools_str = "\n".join([str(tool) for tool in tools])
    return (
        f"{system_prompt}\n\n"
        f"You may use the following tools:\n{tools_str}"
    )

# res = ollama.chat(
#     model="llama3.1",
#     messages=[
#         {"role": "system", "content": get_system_tools_prompt(
#             system_prompt=system_prompt,
#             tools=[search_schema, final_answer_schema]
#         )},
#         # chat history will go here
#         {"role": "user", "content": "hello there"}
#         # scratchpad will go here
#     ],
#     format="json",
# )
# print(res)

import json

# print("\n\n")
# c = json.loads(res["message"]["content"])

# print(c)



# print("\n\n")
# print("pizza now")
# print("\n\n")


# res = ollama.chat(
#     model="llama3.1",
#     messages=[
#         {"role": "system", "content": get_system_tools_prompt(
#             system_prompt=system_prompt,
#             tools=[search_schema, final_answer_schema]
#         )},
#         # chat history will go here
#         {"role": "user", "content": "hi, I'm looking for the best pizzeria in rome"}
#         # scratchpad will go here
#     ],
#     format="json",
# )
# # parse the output
# json.loads(res["message"]["content"])

# print("\n\n")
# print(res)
# print("\n\n")
# c = json.loads(res["message"]["content"])

# print(c)









class AgentAction(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: str | None = None

    @classmethod
    def from_ollama(cls, ollama_response: dict):
        try:
            # parse the output
            output = json.loads(ollama_response["message"]["content"])
            return cls(
                tool_name=output["name"],
                tool_input=output["parameters"],
            )
        except Exception as e:
            print(f"Error parsing ollama response:\n{ollama_response}\n")
            raise e

    def __str__(self):
        text = f"Tool: {self.tool_name}\nInput: {self.tool_input}"
        if self.tool_output is not None:
            text += f"\nOutput: {self.tool_output}"
        return text


# action = AgentAction.from_ollama(res)

# print("\n\n")
# print(action)
# print("\n\n")





def action_to_message(action: AgentAction):
    # create assistant "input" message
    assistant_content = json.dumps({"name": action.tool_name, "parameters": action.tool_input})
    assistant_message = {"role": "assistant", "content": assistant_content}
    # create user "response" message
    user_message = {"role": "user", "content": action.tool_output}
    return [assistant_message, user_message]



# test_action = AgentAction(
#     tool_name="xyz",
#     tool_input={"query": "something cool"},
#     tool_output="A fascinating tidbit of information"
# )
# am = action_to_message(test_action)


# print("\n\n")
# print(am)
# print("\n\n")









def create_scratchpad(intermediate_steps: list[AgentAction]):
    # filter for actions that have a tool_output
    intermediate_steps = [action for action in intermediate_steps if action.tool_output is not None]
    # format the intermediate steps into a "assistant" input and "user" response list
    scratch_pad_messages = []
    for action in intermediate_steps:
        scratch_pad_messages.extend(action_to_message(action))
    return scratch_pad_messages

def call_llm(user_input: str, chat_history: list[dict], intermediate_steps: list[AgentAction]) -> AgentAction:
    # format the intermediate steps into a scratchpad
    scratchpad = create_scratchpad(intermediate_steps)
    # if the scratchpad is not empty, we add a small reminder message to the agent
    if scratchpad:
        scratchpad += [{
            "role": "user",
            "content": (
                f"Please continue, as a reminder my query was '{user_input}'. "
                "Only answer to the original query, and nothing else — but use the "
                "information I provided to you to do so. Provide as much "
                "information as possible in the `answer` field of the "
                "final_answer tool and remember to leave the contact details "
                "of a promising looking restaurant."
            )
        }]
        # we determine the list of tools available to the agent based on whether
        # or not we have already used the search tool
        tools_used = [action.tool_name for action in intermediate_steps]
        tools = []
        if "search" in tools_used:
            # we do this because the LLM has a tendency to go off the rails
            # and keep searching for the same thing
            tools = [final_answer_schema]
            scratchpad[-1]["content"] = " You must now use the final_answer tool."
        else:
            # this shouldn't happen, but we include it just in case
            tools = [search_schema, final_answer_schema]
    else:
        # this would indiciate we are on the first run, in which case we
        # allow all tools to be used
        tools = [search_schema, final_answer_schema]
    # construct our list of messages
    messages = [
        {"role": "system", "content": get_system_tools_prompt(
            system_prompt=system_prompt,
            tools=tools
        )},
        *chat_history,
        {"role": "user", "content": user_input},
        *scratchpad,
    ]
    res = ollama.chat(
        model="llama3.1",
        messages=messages,
        format="json",
    )
    return AgentAction.from_ollama(res)








# out = call_llm(
#     chat_history=[
#         {"role": "user", "content": "hi there, how are you?"},
#         {"role": "assistant", "content": "I'm good, thanks!"},
#         {"role": "user", "content": "I'm currently in Rome"},
#         {"role": "assistant", "content": "That's great, would you like any help?"},
#     ],
#     user_input="yes, I'm looking for the best pizzeria near me",
#     intermediate_steps=[]
# )





# print("\n\nout\n\n")
# print(out)
# print("\n\n")







# if False:
#     results = search(**out.tool_input)
#     print("\n\n")
#     print(results)
#     print("\n\n")












def run_oracle(state: TypedDict):
    print("run_oracle")
    chat_history = state["chat_history"]
    out = call_llm(
        user_input=state["input"],
        chat_history=chat_history,
        intermediate_steps=state["intermediate_steps"]
    )
    return {
        "intermediate_steps": [out]
    }

def router(state: TypedDict):
    print("router")
    # return the tool name to use
    if isinstance(state["intermediate_steps"], list):
        return state["intermediate_steps"][-1].tool_name
    else:
        # if we output bad format go to final answer
        print("Router invalid format")
        return "final_answer"

# we use this to map tool names to tool functions
tool_str_to_func = {
    "search": search,
    "final_answer": final_answer
}

def run_tool(state: TypedDict):
    # use this as helper function so we repeat less code
    tool_name = state["intermediate_steps"][-1].tool_name
    tool_args = state["intermediate_steps"][-1].tool_input
    print(f"run_tool | {tool_name}.invoke(input={tool_args})")
    # run tool
    out = tool_str_to_func[tool_name](**tool_args)
    action_out = AgentAction(
        tool_name=tool_name,
        tool_input=tool_args,
        tool_output=str(out),
    )
    if tool_name == "final_answer":
        return {"output": out}
    else:
        return {"intermediate_steps": [action_out]}







from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

graph.add_node("oracle", run_oracle)
graph.add_node("search", run_tool)
graph.add_node("final_answer", run_tool)

graph.set_entry_point("oracle")  # insert query here

graph.add_conditional_edges(  # - - - >
    source="oracle",  # where in graph to start
    path=router,  # function to determine which node is called
)

# create edges from each tool back to the oracle
for tool_obj in [search_schema, final_answer_schema]:
    tool_name = tool_obj["function"]["name"]
    if tool_name != "final_answer":
        graph.add_edge(tool_name, "oracle")  # ————————>

# if anything goes to final answer, it must then move to END
graph.add_edge("final_answer", END)

runnable = graph.compile()






out = runnable.invoke({
    "input": "where is the best pizza in rome?",
    "chat_history": [],
})




print("\n\n")
print(out["output"]['answer'])
print("\n\n")















