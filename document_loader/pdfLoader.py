from langchain_community.document_loaders import PyPDFLoader

class PDFLoader():
    def __init__(self, path):
        """
        Initialize the PDFLoader

        Args:
            path (str): Path to the PDF file
            
        """
        self.path = path
        self.pdf_loader = PyPDFLoader(self.path)
        self.docs = []

    def loads(self):
        self.docs = self.pdf_loader.load()
        return self.docs