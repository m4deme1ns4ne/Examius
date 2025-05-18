from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from abc import ABC, abstractmethod


class IFileLoader(ABC):
    @abstractmethod
    def load_file():
        pass


class ITextSplitter(ABC):
    @abstractmethod
    def split_text():
        pass


class IDocumentStoreEmbeddings(ABC):
    @abstractmethod
    def create_store():
        pass


class LangChainFileLoader(IFileLoader):
    def __init__(
        self,
        data_for_rag: str,
        show_progress: bool = False,
        which_files_should_add: str = "**/*",
    ):
        self.data_for_rag = data_for_rag
        self.which_files_should_add = which_files_should_add
        self.show_progress = show_progress

    def load_file(self) -> None:
        loader = DirectoryLoader(
            path=self.data_for_rag,
            glob=self.which_files_should_add,
            show_progress=self.show_progress,
        )
        self.documents = loader.load()
        return self.documents


class LangChainTextSplitter(ITextSplitter):
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)


class LangChainOpenAIEmbeddings(IDocumentStoreEmbeddings):
    def create_store(self, split_docs):
        embeddings = OpenAIEmbeddings()
        return Qdrant.from_documents(
            split_docs,
            embeddings,
            location=":memory:",
            collection_name="docs",
        )


class RAGPipeline:
    def __init__(
        self,
        file_loader: IFileLoader,
        text_splitter: ITextSplitter,
        document_store: IDocumentStoreEmbeddings,
    ):
        self.file_loader = file_loader
        self.text_splitter = text_splitter
        self.document_store = document_store

    def execute(self):
        documents = self.file_loader.load_file()
        split_docs = self.text_splitter.split_text(documents)
        return self.document_store.create_store(split_docs)
