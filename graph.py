from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, List
from agents import MultimodalAgent

class GraphState(TypedDict):
    question: str
    image: str | None
    chat_history: List
    answer: str | None

class MultimodalQAGraph:
    def __init__(self, api_key: str):
        """
        Initialize the MultimodalQAGraph.
        
        Args:
            api_key (str): OpenAI API key
        """
        self.agent = MultimodalAgent(api_key)

    def process_query(self, state: GraphState) -> GraphState:
        """
        Process the query using the multimodal agent.
        
        Args:
            state (GraphState): Current state containing question and image
        
        Returns:
            GraphState: Updated state with answer
        """
        response = self.agent.process_query(
            question=state["question"],
            image=state["image"]
        )
        state["answer"] = response
        return state

    def build(self):
        """
        Build the graph workflow.
        
        Returns:
            Compiled graph that can be executed
        """
        workflow = StateGraph(GraphState)
        
        # Add the processing node
        workflow.add_node("process", self.process_query)
        
        # Add edge to END
        workflow.add_edge("process", END)
        
        # Set entry point
        workflow.set_entry_point("process")
        
        return workflow.compile() 