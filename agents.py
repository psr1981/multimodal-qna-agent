from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional
from prompts import MultimodalPromptTemplates

class MultimodalAgent:
    def __init__(self, api_key: str):
        """
        Initialize the MultimodalAgent.
        
        Args:
            api_key (str): OpenAI API key
        """
        self.model = ChatOpenAI(
            model="gpt-4o",
            api_key=api_key,
            max_tokens=1000
        )
        self.prompt_templates = MultimodalPromptTemplates()

    def process_query(self, question: str, image: Optional[str] = None) -> str:
        """
        Process a query with or without an image.
        
        Args:
            question (str): The question to be answered
            image (str, optional): Base64 encoded image
            
        Returns:
            str: Response from the model
        """
        # Get the chat prompt template
        chat_prompt = self.prompt_templates.get_chat_prompt()
        
        # Format the messages
        messages = [
            SystemMessage(content=chat_prompt.messages[0].prompt.template),
            HumanMessage(
                content=self.prompt_templates.format_human_message(question, image)
            )
        ]

        # Get response from the model
        response = self.model.invoke(messages)
        return response.content 