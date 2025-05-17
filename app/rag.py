from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class RAG:
    def __init__(
        self,
        data_for_rag: str,
        show_progress: bool = False,
        which_files_should_add: str = "**/*",
    ) -> None:
        self.data_for_rag = data_for_rag
        self.show_progress = show_progress
        self.which_files_should_add = which_files_should_add

    def file_loader(self) -> None:
        loader = DirectoryLoader(
            path=self.data_for_rag,
            glob=self.which_files_should_add,
            show_progress=self.show_progress,
        )
        self.documents = loader.load()

    def text_splitter(self) -> None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        self.split_docs = text_splitter.split_documents(self.documents)

    def doc_store(self) -> OpenAIEmbeddings:
        self.file_loader()
        self.text_splitter()

        embeddings = OpenAIEmbeddings()
        doc_store = Qdrant.from_documents(
            self.split_docs,
            embeddings,
            location=":memory:",
            collection_name="docs",
        )

        return doc_store
