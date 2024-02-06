import json
from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import BaseLLM
from langchain.pydantic_v1 import Extra
from langchain.schema import LLMResult
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output import GenerationChunk

import os
from datetime import datetime, timedelta
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import requests

# -------------- Env. Variables --------------->>>
CLIENT_ID = os.environ.get("AICORE_OLLAMA_CLIENT_ID")
CLIENT_SECRET = os.environ.get("AICORE_OLLAMA_CLIENT_SECRET")
TOKEN_URL = os.environ.get("AICORE_OLLAMA_AUTH_URL")
API_URL = os.environ.get("AICORE_OLLAMA_API_BASE")
RESOURCE_GROUP = os.environ.get("AICORE_OLLAMA_RESOURCE_GROUP")
DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME")
RETRY_TIME = int(os.environ.get("RETRY_TIME","30"))
DEPLOYMENT_API_PATH = "/lm/deployments" 

# OAuth2 token
TOKEN = {
            "ollama2-server": {
                "envParams": {
                    "token": "AICORE_TOKENURL",
                    "id": "OPENAI_CLIENTID",
                    "sec": "OPENAI_CLIENTSECRET"
                },
                "token": {}
                }
        }

def get_token(service: str) -> str:
    get_token = False
    client = BackendApplicationClient(client_id=CLIENT_ID)
    # create an OAuth2 session
    oauth = OAuth2Session(client=client)
    if TOKEN[service]['token'] == {}:
        get_token = True
    elif datetime.now() > datetime.fromtimestamp(TOKEN[service]['token']['expires_in']):
        get_token = True
    if get_token:
        TOKEN[service]['token'] = oauth.fetch_token(token_url=TOKEN_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
        TOKEN[service]['token']['expires_at'] = datetime.now().timestamp()+TOKEN[service]['token']['expires_in']
    return f"Bearer {TOKEN[service]['token']['access_token']}"
# <<<----------- End of insert -------------------

def _stream_response_to_generation_chunk(
    stream_response: str,
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    parsed_response = json.loads(stream_response)
    generation_info = parsed_response if parsed_response.get("done") is True else None
    return GenerationChunk(
        text=parsed_response.get("response", ""), generation_info=generation_info
    )

class _OllamaCommon(BaseLanguageModel):
# >>>----------- Start of insert -----------------
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
   
    def get_models(self)->list:
        """ Retrieves list of available models in Ollama instance """
        model_url = self.base_url + "/api/tags"
        access_token = get_token(DEPLOYMENT_NAME)
        
        headers = {
            'Authorization': access_token,
            'AI-Resource-Group': RESOURCE_GROUP
        }
        res = requests.get(model_url, headers=headers)
        j_data = res.json()
        return j_data
    
    def pull_model(self, model: str)->list:
        """ Pulls a model through Ollama """
        pull_url = self.base_url + "/api/pull"
        access_token = get_token(DEPLOYMENT_NAME)
        
        headers = {
            'Authorization': access_token,
            'AI-Resource-Group': RESOURCE_GROUP
        }
        result_list = []  # To store the parsed JSON objects
        with requests.post(pull_url, headers=headers, stream=True, json={"name": model}) as res:
            if res.status_code == 200:
                for line in res.iter_lines(chunk_size=1024, decode_unicode=True):
                    if line:
                        # Parse the JSON object from the NDJSON line
                        json_object = json.loads(line)                        
                        # Process or store the parsed JSON object as needed
                        print(f"Model: {model}: {str(json_object)}")                        
                        # Append the parsed JSON object to the result list
                        result_list.append(json_object)
                return result_list
            else:
                # Handle error cases
                print(f"Error: {res.status_code}, {res.text}")    
                return []
# <<<----------- End of insert -------------------
    base_url: str = get_baseurl()
    """Base url the model is hosted under."""

    model: str = "llama2"
    """Model name to use."""

    mirostat: Optional[int] = None
    """Enable Mirostat sampling for controlling perplexity.
    (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"""

    mirostat_eta: Optional[float] = None
    """Influences how quickly the algorithm responds to feedback
    from the generated text. A lower learning rate will result in
    slower adjustments, while a higher learning rate will make
    the algorithm more responsive. (Default: 0.1)"""

    mirostat_tau: Optional[float] = None
    """Controls the balance between coherence and diversity
    of the output. A lower value will result in more focused and
    coherent text. (Default: 5.0)"""

    num_ctx: Optional[int] = None
    """Sets the size of the context window used to generate the
    next token. (Default: 2048)	"""

    num_gpu: Optional[int] = None
    """The number of GPUs to use. On macOS it defaults to 1 to
    enable metal support, 0 to disable."""

    num_thread: Optional[int] = None
    """Sets the number of threads to use during computation.
    By default, Ollama will detect this for optimal performance.
    It is recommended to set this value to the number of physical
    CPU cores your system has (as opposed to the logical number of cores)."""

    repeat_last_n: Optional[int] = None
    """Sets how far back for the model to look back to prevent
    repetition. (Default: 64, 0 = disabled, -1 = num_ctx)"""

    repeat_penalty: Optional[float] = None
    """Sets how strongly to penalize repetitions. A higher value (e.g., 1.5)
    will penalize repetitions more strongly, while a lower value (e.g., 0.9)
    will be more lenient. (Default: 1.1)"""

    temperature: Optional[float] = None
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively. (Default: 0.8)"""

    stop: Optional[List[str]] = None
    """Sets the stop tokens to use."""

    tfs_z: Optional[float] = None
    """Tail free sampling is used to reduce the impact of less probable
    tokens from the output. A higher value (e.g., 2.0) will reduce the
    impact more, while a value of 1.0 disables this setting. (default: 1)"""

    top_k: Optional[int] = None
    """Reduces the probability of generating nonsense. A higher value (e.g. 100)
    will give more diverse answers, while a lower value (e.g. 10)
    will be more conservative. (Default: 40)"""

    top_p: Optional[float] = None
    """Works together with top-k. A higher value (e.g., 0.95) will lead
    to more diverse text, while a lower value (e.g., 0.5) will
    generate more focused and conservative text. (Default: 0.9)"""
    
    images: Optional[List[str]] = None
    """ To submit images data in base64 format to the LLM, e.g. Llava
    """

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Ollama."""
        return {
            "model": self.model,
            "images": self.images,
            "options": {
                "mirostat": self.mirostat,
                "mirostat_eta": self.mirostat_eta,
                "mirostat_tau": self.mirostat_tau,
                "num_ctx": self.num_ctx,
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
                "repeat_last_n": self.repeat_last_n,
                "repeat_penalty": self.repeat_penalty,
                "temperature": self.temperature,
                "stop": self.stop,
                "tfs_z": self.tfs_z,
                "top_k": self.top_k,
                "top_p": self.top_p,
            },
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}
             
    def _create_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop
        elif stop is None:
            stop = []
        params = {**self._default_params, "stop": stop, **kwargs}
        bearer_token = get_token(DEPLOYMENT_NAME)
        response = requests.post(
            url=f"{self.base_url}/api/generate",
            headers={
                "Content-Type": "application/json",
                "AI-Resource-Group": RESOURCE_GROUP,
                "Authorization": bearer_token
            },
            json={"prompt": prompt, **params},
            stream=True,
        )
        print(prompt)   
        response.encoding = "utf-8"
        if response.status_code != 200:
            optional_detail = response.json().get("error")
            raise ValueError(
                f"Ollama call failed with status code {response.status_code}."
                f" Details: {optional_detail}"
            )
        return response.iter_lines(decode_unicode=True)

    def _stream_with_aggregation(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk: Optional[GenerationChunk] = None
        for stream_resp in self._create_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_generation_chunk(stream_resp)
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from Ollama AI Core stream.")

        return final_chunk


class Ollama(BaseLLM, _OllamaCommon):
    """Ollama locally runs large language models.

    To use, follow the instructions at https://ollama.ai/.

    Example:
        .. code-block:: python

            from langchain.llms import Ollama
            ollama = Ollama(model="llama2")
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "ollama-llm"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to Ollama's generate endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = ollama("Tell me a joke.")
        """
        # TODO: add caching here.
        generations = []
        for prompt in prompts:
            final_chunk = super()._stream_with_aggregation(
                prompt,
                stop=stop,
                run_manager=run_manager,
                verbose=self.verbose,
                **kwargs,
            )
            generations.append([final_chunk])
        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for stream_resp in self._create_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_generation_chunk(stream_resp)
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
