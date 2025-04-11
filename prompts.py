from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from typing import Optional

# System prompt template for the multimodal agent
MULTIMODAL_SYSTEM_TEMPLATE = """You are a helpful AI assistant who is trying to help a middle school student to understand concepts and solve problems in easy to understand manner,  
Your role is to:

If Question is from a computational subject or required calculations or proofs or any other math related subject:
    1. If you are responding with math equations then please use latex to format math equations and enclose them in $$. 
    2. Provide step by step solutions to the math related problems
    3. Maintain a professional and helpful tone
    4. recheck your response to make sure that math equations are formatted using latex and enclosed in $$.
else 
    1. Provide clear, concise, and accurate responses
    2. Analyze images when provided and answer questions about them
    3. Be specific and detailed in your observations
    4. Maintain a professional and helpful tone 

Finally, if you're unsure about something in the image, be honest about your uncertainty

"""

# Human prompt template for image analysis
MULTIMODAL_HUMAN_TEMPLATE = """Question: {question}
{image_content}"""

# New DiagramAgent prompts
DIAGRAM_SYSTEM_TEMPLATE = """You are an expert at creating detailed diagram for visualizing the problem. Your role is to:

1. Analyze the given context (text or image)
2. If you believe that diagram can be generated to explain this problem better then 
    a. generate a diagram in form of svg format.
    b. your diagram should contain geometrical shapes, lines and text.
    c. your diagram can contain charts and graphs if you think it is necessary to explain the problem.
3. If your diagram does not contain any shapes, lines, or text then just say nothing, return empty string.
4. If you are not able to generate a diagram then just say nothing, return empty string.
5. Your response must not contain any other text other than generated svg code.

"""

DIAGRAM_HUMAN_TEMPLATE = """Please create a detailed diagram to visualize the problem for the following context:

Context: {context}
{image_content}

Provide a structured diagram that includes:
1. Main components, geometrical shapes
2. Relationships and connections
3. Flow direction
4. Key elements and their attributes"""

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