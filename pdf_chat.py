import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

class PDFChatbot:
    def __init__(self, client, model):
        self.client = client
        self.model = model
        self.vector_store = None
        self.conversation_chain = None
        self.pdf_content = ""

    def process_pdf(self, pdf_file):
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            self.pdf_content = ""
            for page in pdf_reader.pages:
                self.pdf_content += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(self.pdf_content)

            embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"})
            self.vector_store = Chroma.from_texts(chunks, embeddings)

            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.client,
                retriever=self.vector_store.as_retriever(),
                memory=memory
            )
            return True
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return False

    def chat(self, query):
        if not self.conversation_chain:
            return "Please upload a PDF first."

        try:
            response = self.conversation_chain({"question": query})
            return response['result']  # Changed 'answer' to 'result'
        except Exception as e:
            return f"An error occurred while processing your query: {str(e)}"

    def get_pdf_content(self):
        return self.pdf_content