import os
import glob
import pymupdf
from chunking import chunk_text
from vector_store import VectorStore

def extract_clean_text(pdf_path):
    """Extracts and cleans text from a PDF document."""
    doc = pymupdf.open(pdf_path)
    text = "".join([page.get_text() + " " for page in doc])
    return text.replace('\n', ' ').strip()

def main():
    print("Starting PDF processing...")
    pdf_files = glob.glob("data/*.pdf")
    
    chunks = []
    metadatas = []
    
    for pdf_file in pdf_files:
        text = extract_clean_text(pdf_file)
        # Splitting the text into logical chunks
        pieces = chunk_text(text, chunk_size=200, overlap=40)
        
        for i, piece in enumerate(pieces):
            chunks.append(piece)
            metadatas.append({
                "source": os.path.basename(pdf_file),
                "chunk_index": i
            })
            
        print(f"Processed {os.path.basename(pdf_file)}: {len(pieces)} chunks.")

    if not chunks:
        print("No texts found for import. Check the 'data' folder.")
        return

    # Using the provided VectorStore class
    store = VectorStore(name="tech_knowledge_base", reset=True)
    
    print(f"\nImporting {len(chunks)} vectors into ChromaDB. This may take a moment...")
    store.add(chunks=chunks, metadatas=metadatas)
    print(f"Done! The database now contains {store.count()} chunks.")

if __name__ == "__main__":
    main()