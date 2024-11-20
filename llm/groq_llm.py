from langchain_groq import ChatGroq

class GroqLLM():
    def __init__(self, model_name, api_key = None):
        """
        Initialize the GroqLLM

        Args:
            model_name (str): Name of the model
            api_key (str): API Key for the model
        """
        self.model_name = model_name
        self.api_key = api_key
        self.llm = ChatGroq(model_name=self.model_name, api_key=self.api_key)
        
