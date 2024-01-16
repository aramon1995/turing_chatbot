from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from LLMs import TinyLLama
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils import device

llm_dict = {"TinyLLama": TinyLLama}


class Chain:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": device}

    def __init__(self, model="TinyLLama"):
        self.llm_name = model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name, model_kwargs=self.model_kwargs
        )

        self.llm = llm_dict[model]()
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm.llm,
            max_token_limit=512,
            memory_key="chat_history",
            input_key="question",
            return_messages=True,
        )

    def init_chain(self):
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory,
            condense_question_prompt=self.llm.question_generator_template,
            combine_docs_chain_kwargs={"prompt": self.llm.qa_template},
            verbose=True,
            rephrase_question=False,
        )
        
    def load_docs(self, files):
        all_splits = []
        for file in files:
            pdf = "\n".join(
                [page.extract_text() for page in PdfReader(file.name).pages]
            )
            all_splits += self.text_splitter.split_text(pdf)
        # storing embeddings in the vector store
        self.vectorstore = FAISS.from_texts(all_splits, self.embeddings)
        self.init_chain()
        return files

    def response(self, history, question):
        response = self.qa_chain.invoke(question)
        history.append([question, response['answer']])
        return history, ""
