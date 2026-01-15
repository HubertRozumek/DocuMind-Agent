import pdfplumber
from typing import List, Optional, Dict, Union
import os

class PDFLoader:
    """
    Class for loading and extracting text from PDF files.
    """

    def load_pdf(self,filepath: str) -> Dict[str, Union[str, List[str], Dict]]:

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} does not exist")

        try:
            text = ""
            pages_text = []
            metadata = {}

            with pdfplumber.open(filepath) as pdf:
                metadata = pdf.metadata or {}

                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                        pages_text.append(page_text)
                    else:
                        # Fallback
                        page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                        if page_text:
                            text += page_text + "\n\n"
                            pages_text.append(page_text)
                        else:
                            pages_text.append("")
                            print(f"Page {i}: may contain mostly images")

            return {
                "text": text,
                "pages": pages_text,
                "metadata": metadata,
                "page_count": len(pages_text)
            }

        except Exception as e:
            print(f"Exception occurred: {e}")
            raise

    def load_multiple(self,filepaths: List[str]) -> Dict[str, Dict]:
        """
        Load multiple PDF files

        Returns:
            Dict where keys are PDF names and values are PDF files
        """
        results = {}
        for filepath in filepaths:
            try:
                filename = os.path.basename(filepath)
                results[filename] = self.load_pdf(filepath)
                print(f"Loaded {filename} ({results[filename]['page_count']} pages)")
            except Exception as e:
                print(f"Exception occurred: {e} {filepath}")

        return results

    def extract_tables(self, filepaths: str, page_num: Optional[int] = None) -> List:
        """
        Extract tables from PDF files.
        """

        tables = []
        with pdfplumber.open(filepaths) as pdf:
            if page_num is not None:
                pages = [pdf.pages[page_num]]
            else:
                pages = pdf.pages

            for page in pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table:
                        tables.append(table)

        return tables


class DocumentProcessor:
    """
    Menager for document processor.
    """

    def __init__(self, data_dir: str = "data/raw_documents"):
        self.data_dir = data_dir
        self.loader = PDFLoader()

    def get_all_pdfs(self) -> List[str]:
        """
        Return list of all PDF files in directory.
        """
        if not os.path.exists(self.data_dir):
            print(f"Directory {self.data_dir} does not exist")
            return []

        pdf_files = []
        for file in os.listdir(self.data_dir):
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(self.data_dir, file))

        return pdf_files

    def process_all_documents(self) -> Dict:
        """
        Process all PDF files in directory.
        """
        pdf_files = self.get_all_pdfs()

        if not pdf_files:
            print(f"No PDF files in {self.data_dir}")
            return {}

        print(f"Found {len(pdf_files)} PDF files in {self.data_dir}")
        for pdf in pdf_files:
            print(f" - {os.path.basename(pdf)}")

        documents = self.loader.load_multiple(pdf_files)

        total_pages = sum(doc['page_count'] for doc in documents.values())
        total_chars = sum(len(doc['text']) for doc in documents.values())

        print(f" Documents count: {len(documents)}")
        print(f" Total pages: {total_pages}")
        print(f" Total chars: {total_chars}")

        return documents

if __name__ == "__main__":
    processor = DocumentProcessor()
    documents = processor.process_all_documents()