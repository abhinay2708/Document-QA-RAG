from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import GEMINI_MODEL

SYSTEM_PROMPT = """
You are a document question-answering assistant.

Your job is to answer the user's question using
ONLY the information contained in the provided
document context.

Rules:

1. Do not use outside knowledge.
2. Do not make up information.
3. Do not guess.
4. If the answer is not present in the provided
   context, say:

   "I could not find this information in the
   provided documents."

5. Cite the filename and page number for the
   information used.
6. Keep the answer clear and concise.
"""

def get_llm():
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0
    )

    return llm


def get_prompt():
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            SYSTEM_PROMPT
        ),
        (
            "human",
            """
Document Context:

{context}

User Question:

{question}

Answer the question using only the
document context.
"""
        )
    ])

    return prompt