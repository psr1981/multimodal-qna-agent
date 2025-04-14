from langchain_openai import ChatOpenAI
from langchain_community.chat_models import BedrockChat
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional
from prompts import MultimodalPromptTemplates, DiagramPromptTemplates
import boto3
import json

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