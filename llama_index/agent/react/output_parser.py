"""ReAct output parser."""


import re
from typing import Tuple

from llama_index.agent.react.types import (
    Action,
    ActionReasoningStep,
    BaseReasoningStep,
    ResponseReasoningStep,
)
from llama_index.types import BaseOutputParser


def extract_tool_use(input_text: str) -> Tuple[str, str, str]:
    pattern = (
        r"\s*Thought: (.*?)\nAction: ([a-zA-Z0-9_]+).*?\nAction Input: .*?(\{.*?\})"
    )

    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract tool use from input text: {input_text}")

    thought = match.group(1).strip()
    action = match.group(2).strip()
    action_input = match.group(3).strip()
    return thought, action, action_input


def action_input_parser(json_str: str) -> dict:
    processed_string = re.sub(r"(?<!\w)\'|\'(?!\w)", '"', json_str)
    pattern = r'"(\w+)":\s*"([^"]*)"'
    matches = re.findall(pattern, processed_string)
    return dict(matches)


def extract_final_response(input_text: str) -> Tuple[str, str]:
    pattern = r"\s*Gedanke:(.*?)Antwort:(.*?)(?:$)"

    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        raise ValueError(
            f"Could not extract final answer from input text: {input_text}"
        )

    thought = match.group(1).strip()
    answer = match.group(2).strip()
    return thought, answer


class ReActOutputParser(BaseOutputParser):
    """ReAct Output parser."""

    def parse(self, output: str, is_streaming: bool = False) -> BaseReasoningStep:
        """
        Parse output from ReAct agent.

        We expect the output to be in one of the following formats:
        1. If the agent need to use a tool to answer the question:
            ```
            Thought: <thought>
            Action: <action>
            Action Input: <action_input>
            ```
        2. If the agent can answer the question without any tools:
            ```
            Thought: <thought>
            Answer: <answer>
            ```
        """
        if "Werkzeugaufrufe:" in output:
            tool_calls = []
            function_token = "!func"
            regex_tool = rf'(?<={function_token}{{")[^":]+'
            hits_tools = re.findall(regex_tool, output)
            number_of_tool_hits = len(hits_tools)
            hits_tools = list(set(hits_tools))
            number_of_question_hits = 0
            tool_calls = []
            for tool in hits_tools:
                regex_question = rf'(?<={function_token}{{"{tool}": ")[^"]+'
                hits_question = re.findall(regex_question, output)
                for question in hits_question:
                    tool_calls.append((tool, question))
                number_of_question_hits += len(hits_question)
            if number_of_tool_hits != number_of_question_hits:
                raise ValueError("number of tools and questions do not match")
            actions = []
            if len(tool_calls) > 0:
                for tool_call in tool_calls:
                    new_action = Action(
                        thought=f'Calling tool "{tool_call[0]}"',
                        action=tool_call[0],
                        action_input={"input": tool_call[1]},
                    )
                    actions.append(new_action)
                return ActionReasoningStep(actions=actions)
        if "Gedanke:" not in output:
            # NOTE: handle the case where the agent directly outputs the answer
            # instead of following the thought-answer format
            return ResponseReasoningStep(
                thought="(Implizit) Ich kann die Frage beantworten.",
                response=output,
                is_streaming=is_streaming,
            )

        if "Antwort:" in output:
            thought, answer = extract_final_response(output)
            return ResponseReasoningStep(
                thought=thought, response=answer, is_streaming=is_streaming
            )

        raise ValueError(f"Could not parse output: {output}")

    def format(self, output: str) -> str:
        """Format a query with structured output formatting instructions."""
        raise NotImplementedError
