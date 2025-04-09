from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from typing import Optional

# System prompt template for the multimodal agent
MULTIMODAL_SYSTEM_TEMPLATE = """You are a helpful AI assistant who is trying to help a middle school student to understand concepts and solve problems in easy to understand manner,  
Your role is to:

1. If you are responding with math equations then please use latex to format math equations and enclose them in $$. 
2. Provide clear, concise, and accurate responses
3. Analyze images when provided and answer questions about them
4. Be specific and detailed in your observations
5. Provide step by step solutions to the math related problems
6. Maintain a professional and helpful tone
7. If you're unsure about something in the image, be honest about your uncertainty
8. recheck your response to make sure that math equations are formatted using latex and enclosed in $$.


Please provide your analysis and answers based on what you can observe."""

# Human prompt template for image analysis
MULTIMODAL_HUMAN_TEMPLATE = """Question: {question}
{image_content}"""

class MultimodalPromptTemplates:
    @staticmethod
    def get_chat_prompt() -> ChatPromptTemplate:
        """
        Get the complete chat prompt template for multimodal interactions.
        
        Returns:
            ChatPromptTemplate: The configured chat prompt template
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            MULTIMODAL_SYSTEM_TEMPLATE
        )
        
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            MULTIMODAL_HUMAN_TEMPLATE
        )
        
        return ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
    
    @staticmethod
    def format_human_message(question: str, image: Optional[str] = None) -> list:
        """
        Format the human message with question and optional image.
        
        Args:
            question (str): The question to be answered
            image (str, optional): Base64 encoded image data
            
        Returns:
            list: Formatted message content for the API
        """
        if image:
            return [
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image
                    }
                }
            ]
        return question 