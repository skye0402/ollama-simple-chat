from dotenv import load_dotenv
load_dotenv()

from ollama_aicore import ChatOllama

llm = ChatOllama(model="sqlcoder:15b", temperature=0.1)
# llm_response = llm.generate(['Tell me a joke about data scientist', 'Tell me a joke about recruiter', 'Tell me a joke about psychologist'])
# for gen in llm_response.generations:
#     print (gen[0].text)
    
# LangChain supports many other chat models. Here, we're using Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
prompt = ChatPromptTemplate.from_template("Write me a SQL statement about {topic}")

# using LangChain Expressive Language chain syntax
# learn more about the LCEL on
# https://python.langchain.com/docs/expression_language/why
chain = prompt | llm | StrOutputParser()

# for brevity, response is printed in terminal
# You can use LangServe to deploy your application for
# production
print(chain.invoke({"topic": "Retrieving 1 record from table 'test' for all fields in the table without WHERE condition."}))