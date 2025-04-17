from langchain_openai import ChatOpenAI
from langchain_community.chat_models import BedrockChat
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional, Dict, Any
from prompts import MultimodalPromptTemplates, DiagramPromptTemplates
import boto3
import json
from pydantic import BaseModel, Field

class QAResponse(BaseModel):
    """Pydantic model for the QA response format"""
    answer: str = Field(
        description="The comprehensive answer to the question"
    )
    subject: str = Field(
        description="The academic subject this question belongs to (e.g., Mathematics, Physics, Biology, etc.)"
    )

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
            max_tokens=1000,
            model_kwargs={
                "response_format": {"type": "json_object"}
            }
        )
        self.prompt_templates = MultimodalPromptTemplates()

    def _parse_openai_response(self, content: str) -> Dict[str, Any]:
        """
        Parse OpenAI response using Pydantic model.
        
        Args:
            content (str): Raw response content from OpenAI
            
        Returns:
            dict: Validated and parsed response
        """
        try:
            # Clean the input content if needed
            content = content.strip()
            
            # If content is a string representation of JSON, parse it
            if isinstance(content, str):
                content = json.loads(content)
            
            # Validate and parse using Pydantic model
            response = QAResponse(**content)
            
            # Return as dictionary
            return response.model_dump()
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Problematic content: {content}")
            # Return default response if parsing fails
            return QAResponse(
                answer=str(content),
                subject="General"
            ).model_dump()

    def process_query(self, question: str, image: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query with or without an image.
        
        Args:
            question (str): The question to be answered
            image (str, optional): Base64 encoded image
            
        Returns:
            dict: Response from the model with answer and subject
        """
        # Get the chat prompt template
        chat_prompt = self.prompt_templates.get_chat_prompt()
        
        # Create a simplified schema description
        schema_description = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The comprehensive answer to the question"
                },
                "subject": {
                    "type": "string",
                    "description": "The academic subject this question belongs to (e.g., Mathematics, Physics, Biology, etc.)"
                }
            },
            "required": ["answer", "subject"]
        }
        
        # Add the response format to the system message
        system_message = chat_prompt.messages[0].prompt.template
        system_message += f"\n\nYou must respond with a JSON object in the following format:\n{json.dumps(schema_description, indent=2)}"
        
        # Format the messages
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(
                content=self.prompt_templates.format_human_message(question, image)
            )
        ]

        # Get response from the model
        response = self.model.invoke(messages)
        
        # Parse and return the response
        return self._parse_openai_response(response.content)

class DiagramAgent:
    def __init__(self, aws_access_key: str, aws_secret_key: str, region: str, session_token: str = None):
        """
        Initialize the DiagramAgent with AWS Bedrock.
        
        Args:
            aws_access_key (str): AWS access key
            aws_secret_key (str): AWS secret key
            region (str): AWS region
            session_token (str, optional): AWS session token
        """
        #print (aws_access_key, region)
        
        # Initialize AWS Bedrock client
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=session_token,
            region_name=region
        )

       
        # Initialize BedrockChat for Claude 3
        self.model = BedrockChat(
            #model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            #model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
            client=self.bedrock_runtime,
            model_kwargs={
                "max_tokens": 2000,
                "temperature": 0.6
            }
        )
        self.prompt_templates = DiagramPromptTemplates()

    def generate_diagram_description(self, context: str, image: Optional[str] = None) -> str:
        """
        Generate a diagram description using AWS Bedrock's Claude model.
        
        Args:
            context (str): Text description or question
            image (Optional[str]): Base64 encoded image
            
        Returns:
            str: Detailed diagram description
        """
        try:
            chat_prompt = self.prompt_templates.get_diagram_prompt()
            
            messages = [
                SystemMessage(content=chat_prompt.messages[0].prompt.template),
                HumanMessage(content=self.prompt_templates.format_diagram_message(context, image))
            ]

            # Invoke the chat model
            response = self.model.invoke(messages)

            # get the content of the response
            content = response.content.strip()
            print("diagram content", content[0:100])

            # Extract SVG content if present
            if "<svg" in content and "</svg>" in content:
                start_idx = content.find("<svg")
                end_idx = content.find("</svg>") + 6
                content = content[start_idx:end_idx]
            else:
                content = ""

            return content
            
        except Exception as e:
            print(f"Bedrock Error: {str(e)}")
            return f"Error generating diagram description: {str(e)}" 