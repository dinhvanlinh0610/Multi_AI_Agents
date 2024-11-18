from langchain_community.document_loaders import WebBaseLoader

class WebLoader():
    def __init__(self, url):
        self.url = url
        self.web_loader = WebBaseLoader(self.url)
        self.docs = []

    def loads(self):
        self.docs = self.web_loader.load()
        return self.docs
