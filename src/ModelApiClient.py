import json
import re
import os
from dotenv import load_dotenv
import ollama
from openai import OpenAI


class ModelApiClient:
    load_dotenv()

    # Default models for each provider
    MODELS = {
        "openrouter": "deepseek/deepseek-v3.2-exp",
        "local": "mistral-small3.2:24b",
    }
    API_KEYS = {
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
    }
    BASE_URLS = {
        "openrouter": "https://openrouter.ai/api/v1",
        "openai": "https://api.openai.com/v1",
    }

    def __init__(self, provider, model_name=None):
        self.provider = provider
        self.model = model_name if model_name else self.MODELS.get(provider)

        if not self.model:
            raise ValueError(f"Provider '{provider}' model not found")

        if self.provider in ["openrouter", "openai"]:
            self.client = OpenAI(
                api_key=self.API_KEYS[self.provider],
                base_url=self.BASE_URLS[self.provider]
            )

        elif self.provider == "local":
            self.client = None
        else:
            raise ValueError(f"Provider not valid: {self.provider}")

        print(f"ModelApiClient provider: {self.provider}, modell: {self.model}")

    def call(self, messages, **opts):
        if self.provider in ["openrouter"]:
            call_params = {
                "model": self.model,
                "messages": messages,
                "temperature": opts.get("temperature", 1.0),
                "top_p": opts.get("top_p", 1.0),
                "extra_body": {
                    "reasoning": {"enabled": True},
                    "include_reasoning": True
                }
            }
        elif self.provider == "openai":
            # OpenAI rejects 'extra_body' with reasoning params
            call_params = {
                "model": self.model,
                "messages": messages,
                "temperature": opts.get("temperature", 1.0),
                "top_p": opts.get("top_p", 1.0),
            }
        elif self.provider == "local":
            # Handle Local/Ollama separately
            call_params = {
                "model": self.model,
                "messages": messages,
                "options": {"temperature": 1.0, "top_p": 1.0, **opts},
            }
            result = ollama.chat(**call_params)
            return result["message"]["content"]

        response = self.client.chat.completions.create(**call_params)
        message = response.choices[0].message
        content = message.content if message.content else ""

        reasoning = getattr(message, 'reasoning', None) or getattr(message, 'reasoning_content', None)

        if reasoning and "<think>" not in content:
            content = f"<think>{reasoning}</think>\n{content}"

        return content

    @staticmethod
    def split_think_output(s: str):
        flags = re.S | re.I
        thoughts = re.findall(r"<think>(.*?)</think>", s, flags)
        action = re.sub(r"<think>.*?(?:</think>|$)", "", s, flags=flags)
        tool_section = re.sub(r"<(output|o)>.*?</\1>", "", action, flags=flags)
        # Select the final action, never a template/example inside reasoning.
        blocks = re.findall(r"<tool_use>(.*?)</tool_use>(.*?)(?=<tool_use>|$)", tool_section, flags)
        if tool_section.lower().rfind("<tool_use>") > tool_section.lower().rfind("</tool_use>"):
            blocks = []
        tool, tail = blocks[-1] if blocks else ("", "")
        parameters = re.search(r"<parameters>(.*?)</parameters>", tail, flags)
        outputs = re.findall(r"<(output|o)>(.*?)</\1>", action, flags)
        return (thoughts[-1].strip() if thoughts else "", tool.strip(),
                parameters.group(1).strip() if parameters else "",
                outputs[-1][1].strip() if outputs else "")

    @staticmethod
    def extract_tool_info(response_text: str) -> tuple[str, dict | None, str]:
        think, tool_name, param_text, output = ModelApiClient.split_think_output(response_text)

        params = None
        if tool_name and param_text:
            try:
                parsed = json.loads(param_text)
                params = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass

        return tool_name, params, output
