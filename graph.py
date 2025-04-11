from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, List, Any
from agents import MultimodalAgent, DiagramAgent
import asyncio
import concurrent.futures

class GraphState(TypedDict):
    question: str
    image: str | None
    answer: str | None
    diagram: str | None

class MultimodalQAGraph:
    def __init__(self, openai_api_key: str, aws_access_key: str, aws_secret_key: str, 
                 aws_region: str, aws_session_token: str):
        """Initialize the graph with both agents."""
        self.qa_agent = MultimodalAgent(openai_api_key)
        self.diagram_agent = DiagramAgent(
            aws_access_key, 
            aws_secret_key, 
            aws_region,
            aws_session_token
        )

    def process_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input using both agents with true parallelism.
        
        Args:
            state (Dict[str, Any]): Input state containing question and image
            
        Returns:
            Dict[str, Any]: Updated state with both answers
        """
        try:
            question = state["question"]
            image = state["image"]
            
            # Use concurrent.futures to run both agents in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit tasks
                qa_future = executor.submit(
                    self.qa_agent.process_query, 
                    question=question, 
                    image=image
                )
                diagram_future = executor.submit(
                    self.diagram_agent.generate_diagram_description, 
                    context=question, 
                    image=image
                )
                
                # Get results (blocking)
                answer = qa_future.result()
                diagram = diagram_future.result()
            
                #print("answer", answer)
                #print("diagram", diagram)

            # Update state with results
            state["answer"] = answer
            state["diagram"] = diagram
            
            return state
            
        except Exception as e:
            print(f"Error in parallel processing: {str(e)}")
            state["error"] = str(e)
            # Still try to provide something useful if one agent fails
            if not state.get("answer"):
                state["answer"] = "Error processing question"
            if not state.get("diagram"):
                state["diagram"] = "Error generating diagram"
            return state

    def build(self) -> Any:
        """
        Build the graph workflow.
        
        Returns:
            Compiled graph that can be executed
        """
        # Create graph with state definition
        workflow = StateGraph(GraphState)
        
        # Add processing node that handles true parallel execution
        workflow.add_node("process_question", self.process_question)
        
        # Add edge to END
        workflow.add_edge("process_question", END)
        
        # Set entry point
        workflow.set_entry_point("process_question")
        
        return workflow.compile() 
