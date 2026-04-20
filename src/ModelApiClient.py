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
        t = re.search(r"<think>(.*?)</think>", s, flags=re.S | re.I)
        tool = re.search(r"<tool_use>(.*?)</tool_use>", s, flags=re.S | re.I)
        params = re.search(r"<parameters>(.*?)</parameters>", s, flags=re.S | re.I)
        o = re.search(r"<(?:output|o)>(.*?)</(?:output|o)>", s, flags=re.S | re.I)

        think = t.group(1).strip() if t else ""
        tool_use = tool.group(1).strip() if tool else ""
        param_value = params.group(1).strip() if params else ""
        output = o.group(1).strip() if o else ""

        if not output and "</think>" in s:
            remaining = s.split("</think>", 1)[1].strip()
            remaining = re.sub(r"<[^>]+>", "", remaining).strip()
            output = remaining if remaining else s.strip()

        return think, tool_use, param_value, output

    @staticmethod
    def extract_tool_info(response_text: str) -> tuple[str, dict, str]:
        think, tool_name, param_text, output = ModelApiClient.split_think_output(response_text)

        params: dict = {}
        if tool_name and param_text:
            try:
                params = json.loads(param_text)
            except json.JSONDecodeError:
                pass

        return tool_name, params, output
