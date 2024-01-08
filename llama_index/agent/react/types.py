"""Base types for ReAct agent."""

from abc import abstractmethod
from typing import Dict

from llama_index.bridge.pydantic import BaseModel


class BaseReasoningStep(BaseModel):
    """Reasoning step."""

    @abstractmethod
    def get_content(self) -> str:
        """Get content."""

    @property
    @abstractmethod
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""


class Action(BaseModel):
    """Action."""

    thought: str
    action: str
    action_input: Dict


class ActionReasoningStep(BaseReasoningStep):
    """Action Reasoning step."""

    actions: list[Action]

    def get_content(self) -> str:
        """Get content."""
        output = ""
        for action in self.actions:
            output += (
                f"Gedanke: {action.thought}\nAktion: {action.action}\n"
                f"Aktion Input: {action.action_input}\n"
            )
        return output

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False


class ObservationReasoningStep(BaseReasoningStep):
    """Observation reasoning step."""

    observation: str

    def get_content(self) -> str:
        """Get content."""
        return f"Beobachtungen: {self.observation}"

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False


class ResponseReasoningStep(BaseReasoningStep):
    """Response reasoning step."""

    thought: str
    response: str
    is_streaming: bool = False

    def get_content(self) -> str:
        """Get content."""
        if self.is_streaming:
            return (
                f"Thought: {self.thought}\n"
                f"Response (Starts With): {self.response} ..."
            )
        else:
            return f"Gedanke: {self.thought}\n" f"Antwort: {self.response}"

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return True
