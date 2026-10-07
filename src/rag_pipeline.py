from langchain_core.output_parsers import StrOutputParser
from src.retriever import get_retriever
from src.generator import get_llm, get_prompt

def is_greeting(question: str) -> bool:

    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening"
    }

    return question.lower().strip() in greetings

class RAGPipeline:

    def __init__(self):
        print("[RAG] Loading retriever...")

        self.retriever = get_retriever()

        print("[RAG] Loading Gemini...")

        self.llm = get_llm()
        self.prompt = get_prompt()
        self.output_parser = StrOutputParser()

    def format_documents(self, documents):

        formatted = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "Unknown")
            page = document.metadata.get("page")

            if page is not None:
                location = f"{source}, Page {page}"

            else:
                location = source

            formatted.append(
                f"""
SOURCE {index}
Location: {location}

Content:
{document.page_content}
"""
            )

        return "\n".join(formatted)

    def ask(self, question: str):

        # STEP 0: Greeting
        if is_greeting(question):

            return {
                "answer":
                    "Hello! You can ask me questions "
                    "about the documents in my knowledge base.",
                "sources": []
            }
        
        # STEP 1: RETRIEVE

        documents = self.retriever.invoke(question)

        if not documents:
            return {
                "answer":
                    "I could not find this information "
                    "in the provided documents.",
                "sources": []
            }

        # STEP 2: FORMAT CONTEXT

        context = self.format_documents(documents)

        # STEP 3: GENERATE ANSWER

        messages = self.prompt.invoke({
            "context": context,
            "question": question
        })

        response = self.llm.invoke(messages)

        answer = (
            self.output_parser.invoke(
                response
            )
        )

        # STEP 4: SOURCE METADATA

        sources = []
        for document in documents:
            sources.append({
                "source": document.metadata.get("source"),
                "page": document.metadata.get("page"),
                "chunk_id": document.metadata.get("chunk_id"),
                "content": document.page_content
            })

        return {
            "answer": answer,
            "sources": sources
        }