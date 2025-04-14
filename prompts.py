from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from typing import Optional

# System prompt template for the multimodal agent
MULTIMODAL_SYSTEM_TEMPLATE = """You are a helpful AI assistant who is trying to help a middle school student to understand concepts and solve problems in easy to understand manner,  
Your role is to:

If Question is from a computational subject or required calculations or proofs or any other math related subject, then follow these guidelines:
    1. If you are responding with math equations then please use latex to format math equations and enclose equations with $$. 
    2. Provide step by step solutions to the math related problems
    3. Maintain a professional and helpful tone
    4. recheck your response to make sure that they are correct
    5. recheck that math equations are formatted using latex and enclosed in $$.
otherwise, follow these guidelines:
    1. Provide clear, concise, and accurate responses
    2. Analyze images when provided and answer questions about them
    3. Be specific and detailed in your observations
    4. Maintain a professional and helpful tone 

Finally, if you're unsure about something in the image, be honest about your uncertainty

"""

# Human prompt template for image analysis
MULTIMODAL_HUMAN_TEMPLATE = """
Question: {question}
{image_content}

"""

# New DiagramAgent prompts
DIAGRAM_SYSTEM_TEMPLATE = """You are an expert at generating 3d diagram to visualize problems. 

Your role is to:
1. Analyze the given context (text or image)
2. Generate a detailed 3d diagram to explain the problem better.
    a. Generate a 3d diagram in form of svg format.
    b. Your 3d diagram should contain geometrical shapes, lines and text.
    c. Your 3d diagram can also contain charts and graphs if you think it is necessary to explain the problem.
3. If you are not able to generate a 3d diagram then just say nothing, return empty string.
4. Your response must not contain any other text other than generated svg code.
5. your response should be in svg format 

"""

DIAGRAM_HUMAN_TEMPLATE = """

Please generate a 3d diagram to visualize the problem explined in following context:

Context: {context}
{image_content}

"""

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

class DiagramPromptTemplates:
    @staticmethod
    def get_diagram_prompt() -> ChatPromptTemplate:
        """Get the chat prompt template for diagram generation."""
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            DIAGRAM_SYSTEM_TEMPLATE
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            DIAGRAM_HUMAN_TEMPLATE
        )
        return ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
    
    @staticmethod
    def format_diagram_message(context: str, image: Optional[str] = None) -> list:
        """Format the diagram message with context and optional image."""
        if image:
            return [
                {
                    "type": "text",
                    "text": context
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image
                    }
                }
            ]
        return context 