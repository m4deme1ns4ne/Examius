from langchain_openai import OpenAI
from abc import ABC, abstractmethod
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings


class ILanguageModel(ABC):
    @abstractmethod
    def generate_response():
        pass

    @abstractmethod
    def initialisationllm():
        pass


class IMemoryStorage(ABC):
    @abstractmethod
    def add_interaction():
        pass

    @abstractmethod
    def get_history():
        pass


class MemoryManager(IMemoryStorage):
    """Получение сообщений пользователя, ответа LLM при поиске похожих эмбиндингов в RAG"""

    def __init__(self):
        self.memory_list = []

    def add_interaction(self, question: str, answer: str) -> None:
        while len(self.memory_list) > 10:
            self.memory_list.pop(0)
        self.memory_list.append(({"user": question}, {"llm": answer["result"]}))

    def get_history(self) -> list:
        return self.memory_list


class OpenAILanguageModel(ILanguageModel):
    """Объявляем LLM"""

    def __init__(
        self,
        memory: MemoryManager,
        doc_store: OpenAIEmbeddings,
        max_token: int = 100,
        model: str = "gpt-4.1-nano",
    ) -> None:
        """
        Args:
            max_token (int, optional): Максимальная длина ответа в токенах. Defaults to 100.
            model (str, optional): Модель LLM. Defaults to "gpt-4.1-nano".
        """
        self.model = model
        self.max_tokens = max_token
        self.memory = memory
        self.doc_store = doc_store

    def initialisationllm(self) -> None:
        self.llm = OpenAI(model=self.model, temperature=0, max_tokens=self.max_tokens)

    def _qa_chain(self) -> None:
        self.initialisationllm()
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.doc_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            verbose=True,
        )

    def generate_response(self, question: str) -> str:
        """Получение ответа от LLM

        Args:
            question (str): Вопрос пользователя

        Returns:
            str: Ответ LLM
        """
        self._qa_chain()
        try:
            if not question:
                raise ValueError(f"Пустой question.")
            result = self.qa.invoke({"query": question})
            self.memory.add_interaction(question=question, answer=result)
            return result
        except Exception as err:
            print(f"Произошла неизвестная ошибка: {err}")
