import argparse
import traceback

from dotenv import load_dotenv
from langchain import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
# argument parsing logic
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
parser = argparse.ArgumentParser(description="Configure documentation chat system")
parser.add_argument("--url", "-u", type=str, help="URL of the documentation")
parser.add_argument(
    "--model",
    "-m",
    type=str,
    help="LLM to use",
    default="gpt-3.5-turbo",
    choices=["gpt-3.5-turbo", "gpt-4"],
)
parser.add_argument(
    "--temperature", "-t", type=float, help="temperature of the model", default=0.7
)
parser.add_argument(
    "--verbose", "-v", action="store_true", help="increase output verbosity"
)

args = parser.parse_args()

# load the documentation
loader = WebBaseLoader(args.url)
docs = loader.load()
llm = ChatOpenAI(streaming=True, temperature=0, model=args.model)
vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())

chat_history = []

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=args.verbose,
)

# chat loop
print(f"Starting chat session for {args.url} using {args.model} with t={args.temperature}...")
while True:
    try:
        query = input("You: ")
        result = chain({"question": query, "chat_history": chat_history})
        answer = result["answer"]
        print(f"DocGPT: {answer}")
        chat_history += [(query, answer)]
    except KeyboardInterrupt:
        print("Ending chat session...")
        break
    except Exception as e:
        tb = traceback.format_exc()
        print("Oops:", e, tb)
