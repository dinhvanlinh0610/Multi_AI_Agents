from langchain_community.document_loaders import CSVLoader

class CSVLoader():
    def __init__(self, path):
        """
        Initialize the CSVLoader

        Args:
            path (str): Path to the CSV file
            
        """
        self.path = path
        self.csv_loader = CSVLoader(self.path)
        self.docs = []

    def loads(self):
        self.docs = self.csv_loader.load()
        return self.docs