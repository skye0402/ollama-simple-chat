import json
from typing import Any, Iterator, List, Optional

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from ..llm.ollama_aicore import _OllamaCommon
from langchain.schema import ChatResult
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.output import ChatGeneration, ChatGenerationChunk

import os
from datetime import datetime, timedelta
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import requests
# OAuth2 token
token = {
            "ollama2-server": {
                "envParams": {
                    "token": "AICORE_TOKENURL",
                    "id": "OPENAI_CLIENTID",
                    "sec": "OPENAI_CLIENTSECRET"
                },
                "token": {}
                }
        }
# -------------- Env. Variables --------------->>>
CLIENT_ID = os.environ.get("AICORE_LLM_CLIENT_ID")
CLIENT_SECRET = os.environ.get("AICORE_LLM_CLIENT_SECRET")
TOKEN_URL = os.environ.get("AICORE_LLM_AUTH_URL")
API_URL = os.environ.get("AICORE_LLM_API_BASE")
RESOURCE_GROUP = os.environ.get("AICORE_LLM_RESOURCE_GROUP")
DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME")
RETRY_TIME = int(os.environ.get("RETRY_TIME","30"))
DEPLOYMENT_API_PATH = "/lm/deployments"

def get_token(service: str) -> str:
    global token
    client = BackendApplicationClient(client_id=CLIENT_ID)
    # create an OAuth2 session
    oauth = OAuth2Session(client=client)
    if token[service]['token'] == {}:
        token[service]['token'] = oauth.fetch_token(token_url=TOKEN_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)    
    elif datetime.fromtimestamp(token[service]['token']['expires_at']) - datetime.now() < timedelta(seconds=60):
        token[service]['token'] = oauth.fetch_token(token_url=TOKEN_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)  
    return f"Bearer {token[service]['token']['access_token']}"

def get_baseurl()->str:
    """ Retrieves the AI Core deployment URL """
    # Request an access token using client credentials
    access_token = get_token(DEPLOYMENT_NAME)
    
    headers = {
        'Authorization': access_token,
        'AI-Resource-Group': RESOURCE_GROUP
    }
    res = requests.get(API_URL+DEPLOYMENT_API_PATH, headers=headers)
    j_data = res.json()
    for resource in j_data["resources"]:
        if resource["scenarioId"] == DEPLOYMENT_NAME:
            if resource["deploymentUrl"] == "":
                print(f"Scenario '{DEPLOYMENT_NAME}' was found but deployment URL was empty. Current status is '{resource['status']}', target status is '{resource['targetStatus']}'. Retry in {str(RETRY_TIME)} seconds.")
            else:
                print(f"Scenario '{DEPLOYMENT_NAME}': Plan '{resource['details']['resources']['backend_details']['predictor']['resource_plan']}', modfied at {resource['modifiedAt']}.")
            return f"{resource['deploymentUrl']}/v1"

def _stream_response_to_chat_generation_chunk(
    stream_response: str,
) -> ChatGenerationChunk:
    """Convert a stream response to a generation chunk."""
    parsed_response = json.loads(stream_response)
    generation_info = parsed_response if parsed_response.get("done") is True else None
    return ChatGenerationChunk(
        message=AIMessageChunk(content=parsed_response.get("response", "")),
        generation_info=generation_info,
    )


class ChatOllama(BaseChatModel, _OllamaCommon):
    """Ollama locally runs large language models.

    To use, follow the instructions at https://ollama.ai/.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatOllama
            ollama = ChatOllama(model="llama2")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "ollama-chat"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    def _format_message_as_text(self, message: BaseMessage) -> str:
        if isinstance(message, ChatMessage):
            message_text = f"\n\n{message.role.capitalize()}: {message.content}"
        elif isinstance(message, HumanMessage):
            message_text = f"[INST] {message.content} [/INST]"
        elif isinstance(message, AIMessage):
            message_text = f"{message.content}"
        elif isinstance(message, SystemMessage):
            message_text = f"<<SYS>> {message.content} <</SYS>>"
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _format_messages_as_text(self, messages: List[BaseMessage]) -> str:
        return "\n".join(
            [self._format_message_as_text(message) for message in messages]
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to Ollama's generate endpoint.

        Args:
            messages: The list of base messages to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            Chat generations from the model

        Example:
            .. code-block:: python

                response = ollama([
                    HumanMessage(content="Tell me about the history of AI")
                ])
        """

        prompt = self._format_messages_as_text(messages)
        final_chunk = super()._stream_with_aggregation(
            prompt, stop=stop, run_manager=run_manager, verbose=self.verbose, **kwargs
        )
        chat_generation = ChatGeneration(
            message=AIMessage(content=final_chunk.text),
            generation_info=final_chunk.generation_info,
        )
        return ChatResult(generations=[chat_generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt = self._format_messages_as_text(messages)
        for stream_resp in self._create_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_chat_generation_chunk(stream_resp)
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
