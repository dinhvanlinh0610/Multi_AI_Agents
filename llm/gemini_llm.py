import google.generativeai as genai 

class GeminiLLM():
    def __init__(self, model_name, api_key = None):
        """
        Initialize the GeminiLLM

        Args:
            model_name (str): Name of the model
            api_key (str): API Key for the model
        """
        self.model_name = model_name
        self.api_key = api_key
        if api_key:
            genai.configure(api_key=self.api_key)
        self.llm = genai.GenerativeModel(model_name)
        
