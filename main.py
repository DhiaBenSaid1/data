from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from langchain_core.output_parsers import StrOutputParser

# Initialize the model
model = OllamaLLM(model="llama3.2")

# Prompt template for train and line data
template = """
You are an expert in answering questions about train and line data.

Here are some relevant train and line records: {records}

Here is the question to answer: {question}

Answer:
"""

# Create prompt and chain
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model | StrOutputParser()

# Interactive loop
print("Welcome to the Train Data Assistant!")
while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question.lower() == "q":
        break
    
    records = "\n".join([doc.page_content for doc in retriever.invoke(question)])
    result = chain.invoke({"records": records, "question": question})
    print("Answer:")
    print(result)