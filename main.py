import os
from app.config import Config
from app.llm import OpenAILanguageModel, MemoryManager
from app.rag import RAG
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA


def main():

    load_dotenv(find_dotenv(raise_error_if_not_found=True))

    config = Config(
        PROXY=os.getenv("PROXY"), OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    )

    rag = RAG(data_for_rag="data_for_preprocess", show_progress=True)

    doc_store = rag.doc_store()

    memory_manager = MemoryManager()

    llm = OpenAILanguageModel(memory_manager, doc_store)

    # Цикл запросов
    while True:
        try:
            question = input("Ваш вопрос: ")
            if not question:
                continue
            result = llm.generate_response(question)
            print(f"Ответ: {result["result"]}")
        except Exception as err:
            print(f"Произошла ошибка: {err}")
        except KeyboardInterrupt:
            print("\nЗавершение программы...")
            break


if __name__ == "__main__":
    main()
