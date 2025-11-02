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


class ComparisonAgent:
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model
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

        # Extract text from message
        user_text = ""
        for part in user_message.parts:
            if part.kind == "text":
                user_text = part.text.strip()
                break

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
                    if "insight" in content:
                        comparison_data = content

                messages.append(message_obj)
                messages.extend(results)
            else:
                done = True
                assistant_message = choice.message
                messages.append(assistant_message)

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
