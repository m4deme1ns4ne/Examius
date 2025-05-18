import os
from app.utils.config import Config
from app.core.llm import OpenAILanguageModel, MemoryManager
from app.core.rag import (
    LangChainFileLoader,
    LangChainTextSplitter,
    LangChainOpenAIEmbeddings,
    RAGPipeline,
)
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Depends, HTTPException
from app.models.models import Question, Response
import uvicorn
from contextlib import asynccontextmanager
from loguru import logger

load_dotenv(find_dotenv(raise_error_if_not_found=True))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Инициализация приложения...")
    try:
        config = Config(
            PROXY=os.getenv("PROXY"), OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
        )
        logger.debug("Конфигурация загружена")

        # Инициализация компонентов
        logger.info("Инициализация компонентов RAG...")
        file_loader = LangChainFileLoader(data_for_rag="data_for_rag", show_progress=True)
        text_splitter = LangChainTextSplitter()
        document_store = LangChainOpenAIEmbeddings()

        # Сборка пайплайна
        logger.info("Запуск RAG пайплайна...")
        pipeline = RAGPipeline(file_loader, text_splitter, document_store)
        doc_store = pipeline.execute()
        logger.info("RAG пайплайн успешно выполнен")

        memory_manager = MemoryManager()
        llm = OpenAILanguageModel(memory_manager, doc_store)

        app.state.doc_store = doc_store
        app.state.memory_manager = memory_manager
        app.state.config = config
        logger.info("Приложение успешно инициализировано")
    except Exception as e:
        logger.error(f"Ошибка при инициализации приложения: {str(e)}")
        raise

    yield


app = FastAPI(
    title="API для получения ответа от LLM вместе с RAG",
    description="API для получения ответа от LLM вместе с RAG и историей сообщений.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", summary="Проверка состояния")
def health_check():
    logger.debug("Получен запрос на проверку состояния")
    return {"status": "ok"}


def get_llm():
    return OpenAILanguageModel(app.state.memory_manager, app.state.doc_store)


@app.post("/ask", summary="Получения ответа от LLM")
def ask_question(question: Question, llm=Depends(get_llm)):
    logger.info(f"Получен новый вопрос: {question.question[:100]}...")
    try:
        result = llm.generate_response(
            f"Запрос пользователя: {question.question}. Предыдущие сообщения: {app.state.memory_manager.get_history()}"
        )
        logger.debug("Ответ успешно сгенерирован")
        return Response(
            answer=result["result"], history=app.state.memory_manager.get_history()
        )
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка при генерации ответа")


def main():
    logger.info("Запуск сервера на порту 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
