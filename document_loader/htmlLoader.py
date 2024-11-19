from langchain_community.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader

class HTMLLoader():

    def __init__(self, path, loader_type="unstructured"):
        """
        Initialize the HTMLLoader

        Args:
            path (str): Path to the HTML file
            
        """
        self.path = path
        if loader_type == "unstructured":
            self.html_loader = UnstructuredHTMLLoader(self.path)
        else:
            self.html_loader = BSHTMLLoader(self.path)
        self.docs = []

    def loads(self):
        self.docs = self.html_loader.load()
        return self.docs