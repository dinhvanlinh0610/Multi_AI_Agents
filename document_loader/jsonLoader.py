from langchain_community.document_loaders import JSONLoader

class JSLoader():

    def __init__(self, path, json_lines=False, jq_schema="."):
        """
        Initialize the JSLoader

        Args:
            path (str): Path to the JSON file
            
        """
        self.path = path
        self.json_loader = JSONLoader(
            file_path=self.path,
            jq_schema=jq_schema,
            json_lines=json_lines,
            text_content=False
            )
        self.docs = []

    def loads(self):
        self.docs = self.json_loader.load()
        return self.docs
