import os
import re
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# LOAD ENV VARIABLES
load_dotenv()

# STREAMLIT CONFIG
st.set_page_config(page_title=" Smart Document QA Assistant", layout="wide")
st.title(" Smart Document QA Assistant")
st.caption("Ask questions about any uploaded document using a Retrieval-Augmented Generation (RAG) pipeline ")


# FUNCTIONS
def clean_text(docs):
    """Clean non-UTF characters and extra spaces from documents."""
    for doc in docs:
        doc.page_content = re.sub(r'[^\x00-\x7F]+', ' ', doc.page_content)
        doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()
    return docs


def format_docs(retrieved_docs):
    """Format retrieved documents into a single text block."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


# STREAMLIT UI
uploaded_file = st.file_uploader(" Upload a PDF or Text Document", type=["pdf", "txt"])
query = st.text_input(" Ask a question:")
debug_mode = st.toggle(" Debug Mode (show retrieved context)", value=False)

if uploaded_file:
    with st.spinner(" Processing your document..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        # LOAD DOCUMENT
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        else:
            loader = TextLoader(temp_file_path)
        docs = loader.load()

        # Clean weird symbols
        docs = clean_text(docs)

        # SPLIT INTO CHUNKS
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(docs)

        # CREATE EMBEDDINGS & VECTORSTORE
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # DEFINE PROMPT + LLM
        prompt = PromptTemplate(
            template="""
            You are a helpful AI assistant.
            Use ONLY the following context to answer the question.
            If the answer cannot be found in the context, say "I don’t know."

            Context:
            {context}

            Question: {question}

            Answer:
            """,
            input_variables=["context", "question"]
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


        # BUILD RAG CHAIN
        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser

        st.success(" Document processed successfully!")

        # HANDLE USER QUERY
        if query:
            with st.spinner(" Thinking..."):
                answer = main_chain.invoke(query)

            st.markdown("###  Answer:")
            st.write(answer.strip())

            # Optional Debug Mode
            if debug_mode:
                st.divider()
                st.markdown("### Retrieved Context (Debug Info)")
                context_docs = retriever.invoke(query)
                for i, doc in enumerate(context_docs, start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(doc.page_content)
                    st.divider()

else:
    st.info(" Please upload a document to begin.")
