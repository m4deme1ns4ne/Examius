import os
from app.config import Config
from app.llm import OpenAILanguageModel, MemoryManager
from app.rag import (
    LangChainFileLoader,
    LangChainTextSplitter,
    LangChainOpenAIEmbeddings,
    RAGPipeline,
)
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

load_dotenv(find_dotenv(raise_error_if_not_found=True))

config = Config(
    PROXY=os.getenv("PROXY"), OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
)

# Инициализация компонентов
file_loader = LangChainFileLoader(
    data_for_rag="data_for_rag", show_progress=True
)
text_splitter = LangChainTextSplitter()
document_store = LangChainOpenAIEmbeddings()

# Сборка пайплайна
pipeline = RAGPipeline(file_loader, text_splitter, document_store)
doc_store = pipeline.execute()

memory_manager = MemoryManager()
llm = OpenAILanguageModel(memory_manager, doc_store)


class Question(BaseModel):
    question: str


app = FastAPI(
    title="API для получения ответа от LLM вместе с RAG",
    description="API для получения ответа от LLM вместе с RAG и историей сообщений.",
    version="1.0.0"
)

@app.get("/health", summary="Проверка состояния")
def health_check():
    return {
        "status": "ok"
    }

@app.post("/ask", summary="Получения ответа от LLM")
def ask_question(question: Question):
    llm = OpenAILanguageModel(memory_manager, doc_store)
    result = llm.generate_response(
        f"Запрос пользователя: {question}. Предыдущие сообщения: {memory_manager.get_history()}"
    )
    return result["result"], memory_manager.get_history()


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
