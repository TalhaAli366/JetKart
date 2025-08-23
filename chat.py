from langchain.chat_models import init_chat_model
llm = init_chat_model("google_genai:gemini-2.5-flash")
output = llm.invoke("hello")
print(output.response)
