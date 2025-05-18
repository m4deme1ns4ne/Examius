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


def main():

    load_dotenv(find_dotenv(raise_error_if_not_found=True))

    config = Config(
        PROXY=os.getenv("PROXY"), OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    )

    # Инициализация компонентов
    file_loader = LangChainFileLoader(
        data_for_rag="data_for_preprocess", show_progress=True
    )
    text_splitter = LangChainTextSplitter()
    document_store = LangChainOpenAIEmbeddings()

    # Сборка пайплайна
    pipeline = RAGPipeline(file_loader, text_splitter, document_store)
    doc_store = pipeline.execute()

    memory_manager = MemoryManager()
    llm = OpenAILanguageModel(memory_manager, doc_store)

    # Цикл запросов
    while True:
        try:
            question = input("Ваш вопрос: ")
            if not question:
                continue
            result = llm.generate_response(
                f"Запрос пользователя: {question}. Предыдущие сообщения: {memory_manager.get_history()}"
            )
            print(f"Ответ: {result["result"]}")
            print(f"\n{memory_manager.get_history()}")
        except Exception as err:
            print(f"Произошла ошибка: {err}")
        except KeyboardInterrupt:
            print("\nЗавершение программы...")
            break


if __name__ == "__main__":
    main()
