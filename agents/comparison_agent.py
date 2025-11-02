from uuid import uuid4
from typing import List, Optional
import json
from openai import AsyncOpenAI

from models.a2a import (
    A2AMessage, TaskResult, TaskStatus, Artifact,
    MessagePart, MessageConfiguration
)
from utils.utils import (
    compare_companies, handle_tool_calls, 
    is_comparison_query, tools
)
from utils.data import system_prompt
import os
from dotenv import load_dotenv

load_dotenv()


class ComparisonAgent:
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=os.getenv("GOOGLE_API_KEY"),
                                  base_url=os.getenv("BASE_URL"))
        self.model = os.getenv('MODEL')
        self.conversations = {}  # Store conversation history by context_id

    async def process_messages(
        self,
        messages: List[A2AMessage],
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
        config: Optional[MessageConfiguration] = None
    ) -> TaskResult:
        """Process incoming messages and generate company comparisons"""
        
        # Generate IDs if not provided
        context_id = context_id or str(uuid4())
        task_id = task_id or str(uuid4())
        
        # Get or create conversation history for this context
        history = self.conversations.get(context_id, [])
        
        # Extract last user message
        user_message = messages[-1] if messages else None
        if not user_message:
            raise ValueError("No message provided")
        
        # Extract text from message parts (including nested data)
        user_text = self._extract_latest_text(user_message)
        
        if not user_text:
            raise ValueError("No text content found in message")
        
        # Check if it's a comparison query
        if not is_comparison_query(user_text):
            error_msg = (
                "This AI Agent is designed only for comparing two companies.\n\n"
                "Example inputs:\n"
                "- Compare Apple and Microsoft\n"
                "- How does Tesla compare to Ford?\n"
                "- Which is better, Google or Amazon?\n\n"
                "Please rephrase your request to include two companies for comparison."
            )
            
            response_message = A2AMessage(
                role="agent",
                parts=[MessagePart(kind="text", text=error_msg)],
                taskId=task_id
            )
            
            return TaskResult(
                id=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state="completed",
                    message=response_message
                ),
                artifacts=[],
                history=messages + [response_message]
            )
        
        # Process with OpenAI and tools
        try:
            assistant_response, comparison_data = await self._chat_with_tools(
                user_text, 
                history
            )
            
            # Update conversation history
            self.conversations[context_id] = history
            
            # Build response message
            response_message = A2AMessage(
                role="agent",
                parts=[MessagePart(kind="text", text=assistant_response)],
                taskId=task_id
            )
            
            # Build artifacts
            artifacts = []
            
            # Add text response artifact
            artifacts.append(
                Artifact(
                    name="comparison_text",
                    parts=[MessagePart(kind="text", text=assistant_response)]
                )
            )
            
            # Add structured data artifact if available
            if comparison_data:
                artifacts.append(
                    Artifact(
                        name="comparison_data",
                        parts=[MessagePart(kind="data", data=comparison_data)]
                    )
                )
            
            # Build full history
            full_history = messages + [response_message]
            
            return TaskResult(
                id=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state="completed",
                    message=response_message
                ),
                artifacts=artifacts,
                history=full_history
            )
            
        except Exception as e:
            error_msg = f"An error occurred while processing your request: {str(e)}"
            
            response_message = A2AMessage(
                role="agent",
                parts=[MessagePart(kind="text", text=error_msg)],
                taskId=task_id
            )
            
            return TaskResult(
                id=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state="failed",
                    message=response_message
                ),
                artifacts=[],
                history=messages + [response_message]
            )

    def _extract_latest_text(self, message: A2AMessage) -> str:
        """
        Extract the latest text from message parts.
        Handles nested data arrays with historical messages.
        """
        latest_text = ""
        
        for part in message.parts:
            if part.kind == "text" and part.text:
                # Direct text in the main parts
                latest_text = part.text.strip()
            elif part.kind == "data" and part.data:
                # Check if data is a list (history)
                if isinstance(part.data, list):
                    # Get the last text message from history
                    for item in reversed(part.data):
                        if isinstance(item, dict) and item.get("kind") == "text":
                            text = item.get("text", "")
                            # Skip error messages and HTML tags
                            if text and not text.startswith("sorry") and not text.startswith("<"):
                                latest_text = text.strip()
                                break
        
        return latest_text

    async def _chat_with_tools(self, message: str, history: list) -> tuple[str, dict]:
        """
        Chat with OpenAI using tools for company comparison.
        Returns (assistant_response, comparison_data)
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})
        
        comparison_data = None
        done = False
        
        while not done:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools
            )
            
            choice = response.choices[0]
            finish_reason = choice.finish_reason
            
            if finish_reason == 'tool_calls':
                message_obj = choice.message
                tool_calls = message_obj.tool_calls
                results = handle_tool_calls(tool_calls)
                
                # Extract comparison data from tool results
                for result in results:
                    content = json.loads(result["content"])
                    if "insight" in content or "company1" in content:
                        comparison_data = content
                
                # Convert message_obj to dict for messages list
                messages.append({
                    "role": "assistant",
                    "content": message_obj.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in tool_calls
                    ]
                })
                messages.extend(results)
            else:
                done = True
                assistant_message = choice.message
                
                # Update history
                history.append({"role": "user", "content": message})
                history.append({
                    "role": "assistant", 
                    "content": assistant_message.content
                })
                
                return assistant_message.content, comparison_data

    async def cleanup(self):
        """Cleanup resources"""
        self.conversations.clear()
        await self.client.close()