"""ReAct output parser."""


import json
import re
from typing import Tuple

from llama_index.agent.react.types import (
    Action,
    ActionReasoningStep,
    BaseReasoningStep,
    ResponseReasoningStep,
)
from llama_index.output_parsers.utils import extract_json_str
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
    pattern = r"\s*Thought:(.*?)Answer:(.*?)(?:$)"

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
        if "Tool calls:" in output:
            print("checking for tools")
            output_to_test = output.replace(r"\_", "_")
            tool_calls = []
            function_token = "!functioncall"
            regex_tool = rf'(?<={function_token}\[")[^":]+'
            hits_tools = re.findall(regex_tool, output_to_test)
            hits_questions = []
            print("hits_tools", hits_tools)
            for tool in hits_tools:
                regex_question = rf'(?<={function_token}\["{tool}": ")[^"]+'
                hits_question = re.findall(regex_question, output_to_test)
                hits_questions += hits_question
            if len(hits_tools) == len(hits_questions):
                for i in range(len(hits_tools)):
                    tool = hits_tools[i]
                    question = hits_questions[i]
                    print("tool added", tool, question)
                    tool_calls.append((tool, question))
            if len(hits_tools) != len(hits_questions):
                raise ValueError("number of tools and questions do not match")
            actions = []
            if len(tool_calls) > 0:
                for tool_call in tool_calls:
                    new_action = Action(
                        thought=f"Calling data repository {tool_call[0]}",
                        action=tool_call[0],
                        action_input={"input": tool_call[1]},
                    )
                    actions.append(new_action)
                return ActionReasoningStep(actions=actions)
        if "Thought:" not in output:
            # NOTE: handle the case where the agent directly outputs the answer
            # instead of following the thought-answer format
            return ResponseReasoningStep(
                thought="(Implicit) I can answer without any more tools!",
                response=output,
                is_streaming=is_streaming,
            )

        if "Answer:" in output:
            thought, answer = extract_final_response(output)
            return ResponseReasoningStep(
                thought=thought, response=answer, is_streaming=is_streaming
            )

        if "Action:" in output:
            thought, action, action_input = extract_tool_use(output)
            json_str = extract_json_str(action_input)

            # First we try json, if this fails we use ast
            try:
                action_input_dict = json.loads(json_str)
            except json.JSONDecodeError:
                action_input_dict = action_input_parser(json_str)

            return ActionReasoningStep(
                thought=thought, action=action, action_input=action_input_dict
            )

        raise ValueError(f"Could not parse output: {output}")

    def format(self, output: str) -> str:
        """Format a query with structured output formatting instructions."""
        raise NotImplementedError
