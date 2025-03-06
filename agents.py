from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional

class MultimodalAgent:
    def __init__(self, api_key: str):
        """
        Initialize the MultimodalAgent.
        
        Args:
            api_key (str): OpenAI API key
        """
        # Simplified configuration without extra parameters
        self.model = ChatOpenAI(
            model="gpt-4o",
            api_key=api_key,
            max_tokens=1000
        )

    def process_query(self, question: str, image: Optional[str] = None) -> str:
        """
        Process a query with or without an image.
        
        Args:
            question (str): The question to be answered
            image (str, optional): Base64 encoded image
            
        Returns:
            str: Response from the model
        """
        messages = [
            SystemMessage(content=(
                "You are a helpful assistant that can analyze images and answer questions about them. "
                "Provide clear, concise, and accurate responses."
            ))
        ]

        if image:
            # If image is provided, create a message with image content
            message_content = [
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                }
            ]
            messages.append(HumanMessage(content=message_content))
        else:
            # If no image, just process the text question
            messages.append(HumanMessage(content=question))

        # Get response from the model
        response = self.model.invoke(messages)
        return response.content 