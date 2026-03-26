import os
from PyPDF2 import PdfReader
#from pdf2image import convert_from_path
#import pytesseract
#import requests
import pickle
#import sys
#import tempfile
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()

class PDFIndexer:
    def __init__(self, text_output_dir='extracted_pdf_texts'):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.text_output_dir = os.path.join(current_file_dir,text_output_dir)
        self.documents = []
        
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("API_KEY"), model="text-embedding-3-large")
        os.makedirs(self.text_output_dir, exist_ok=True)

    def load_documents_from_folder(self, folder_path, use_pkl=False):

        if use_pkl:

            print("Loading documents from pickle file")
            with open(f"{folder_path}/documents.pkl", "rb") as file:
                self.documents = pickle.load(file)

        else:
            print(f"Loading text documents from {folder_path}")
            documents = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, "r", encoding='utf-8') as file:
                        text = file.read()
                        chunks = text_splitter.split_text(text)
                        for chunk in chunks:
                            chunk_metadata = {"filename": filename, "chunk_start": chunk[:50]}
                            doc = Document(page_content=chunk, metadata=chunk_metadata)
                            documents.append(doc)
            self.documents = documents
            self.save_documents_to_pkl(self.documents, pkl_folder="pickled_rag_objects")
    
    def normalize_embeddings(self, embeddings):
        """Normalize embeddings to unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def extract_text_from_pdf(self, folder_path, title):
        reader = PdfReader(f"{folder_path}/{title}")
        num_pages = len(reader.pages)
        full_text = ""

        print("number of pages", num_pages)

        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text = page.extract_text() or ""
            
            # images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=300)
            # if images:
            #     image = images[0]
            #     ocr_text = pytesseract.image_to_string(image)
            #     text += "\n" + ocr_text

            metadata = {
                "textbook_name": title[:-3],
                "page_number": page_num + 1
            }
            self.store_text_with_metadata(text, metadata)
            print("curr length of documents", len(self.documents), page_num+1)
            full_text += text

        return full_text
    
    def store_text_with_metadata(self, text, metadata):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)

        for chunk in chunks:
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_start"] = chunk[:50]  # Add a snippet of the chunk start for reference
            doc = Document(page_content=chunk, metadata=chunk_metadata)
            self.documents.append(doc)

    def process_pdfs(self, folder_path):
        # Extract text from PDF
        for title in os.listdir(folder_path):
            text = self.extract_text_from_pdf(folder_path, title)
            text_file_path = os.path.join(self.text_output_dir, f"{title}.txt")
            try:
                with open(text_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(text)
                print(f"Saved extracted text to {text_file_path}")
            except Exception as e:
                print(f"Failed to save text for {title}: {str(e)}")

    def create_vector_store(self, use_pkl=False):
        
        pkl_folder = "pickled_rag_objects"
        if use_pkl:
            print("Loading embeddings from pickle file")
            embeddings_list = []
            with open(f"{pkl_folder}/embeddings.pkl", "rb") as file:
                embeddings_list = pickle.load(file)
        
        else:
            # Generate embeddings for all documents
            print("Generating embeddings")
            embeddings_list = self.embeddings.embed_documents([d.page_content for d in self.documents])
            self.save_embedded_documents_to_pkl(embeddings_list, pkl_folder=pkl_folder)

        # Normalize the embeddings
        print("Normalizing embeddings")
        print(len(np.array(embeddings_list)))
        normalized_embeddings = self.normalize_embeddings(np.array(embeddings_list))
        
        # Create FAISS index with Inner Product (IP)
        print("Creating FAISS index")
        dimension = normalized_embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(normalized_embeddings)

        docstore = InMemoryDocstore({
            str(i): self.documents[i]
            for i in range(len(self.documents))
        })
        
        # Create FAISS vectorstore from the index
        print("Creating FAISS vectorstore")
        print(len(self.documents))
        vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=faiss_index,
            docstore=docstore,
            index_to_docstore_id={i: str(i) for i in range(len(self.documents))},
        )
        vectorstore.save_local("pdf_vector_store")
        print("Vector store created and saved locally.")
    
    def save_embedded_documents_to_pkl(self, embeddings, pkl_folder):
        with open(f"{pkl_folder}/embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved in {pkl_folder}")
    
    def save_documents_to_pkl(self, documents, pkl_folder):
        with open(f"{pkl_folder}/documents.pkl", "wb") as f:
            pickle.dump(documents, f)
        print(f"Documents saved in {pkl_folder}")

    def add_new_pdf(self, folder_path, title):
        """
        Process a new PDF file given its URL and title:
        1. Extract text from the PDF.
        2. Split the text into chunks (if your use-case requires segmentation).
        3. Create Document objects with metadata.
        4. Load the existing FAISS vector store.
        5. Add the new documents to the index and save it locally.
        """
        # Extract text from the new PDF file
        original_len = len(self.documents)
        print("original length of documents", original_len)
        text = self.extract_text_from_pdf(folder_path, title)
        text_file_path = os.path.join(self.text_output_dir, f"{title}.txt")
        print("length of documents", len(self.documents))
        try:
            with open(text_file_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text)
            print(f"Saved extracted text to {text_file_path}")
        except Exception as e:
            print(f"Failed to save text for {title}: {str(e)}")

        new_documents = self.documents[original_len:]

        current_dir = os.path.dirname(os.path.abspath(__file__))
        documents_dir = os.path.join(current_dir, "document_pkls")
        os.makedirs(documents_dir, exist_ok=True)

        pkl_filename = "new_documents.pkl"
        pkl_path = os.path.join(documents_dir, pkl_filename)
        
        # Save the new documents
        with open(pkl_path, "wb") as f:
            pickle.dump(new_documents, f)
        print(f"Saved {len(new_documents)} new document chunks to {pkl_path}")

        if new_documents:
          try:
              # Load existing vector store
              current_dir = os.path.dirname(os.path.abspath(__file__))
              vector_store_path = os.path.join(current_dir, "pdf_vector_store")
              vector_store = FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
              
              # Add new documents to the vector store
              vector_store.add_documents(new_documents)
              
              # Save updated vector store
              vector_store.save_local("pdf_vector_store")
              print(f"Added new PDF '{title}' to the vector store.")
          except Exception as e:
              print(f"Error updating vector store: {str(e)}")


if __name__ == "__main__":
    indexer = PDFIndexer()


    # # Uncomment the below code when new PDFs are added to the database

    # for textbook in textbooks:
    #     print("textbook",textbook["title"], textbook["textbook_url"])
    #     indexer.process_pdfs(textbook['textbook_url'], textbook['title'])

    # indexer.create_vector_store()

    # Comment the above code and uncomment the below code to add a new PDF

    pkl_folder = "pickled_rag_objects"
    text_folder = "extracted_pdf_texts"

    #indexer.process_pdfs(folder_path="papers_for_rag")
    indexer.load_documents_from_folder(folder_path=pkl_folder, use_pkl=True)
    indexer.create_vector_store()

    print("Pdf vector store created")
