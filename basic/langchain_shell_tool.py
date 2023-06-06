from langchain.tools import ShellTool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType

llm = ChatOpenAI(temperature=0)
shell_tool = ShellTool()

shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace("{", "{{").replace("}", "}}")
self_ask_with_search = initialize_agent([shell_tool], llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
self_ask_with_search.run("Download the langchain.com webpage and grep for all urls. Return only a sorted list of them. Be sure to use double quotes.")
