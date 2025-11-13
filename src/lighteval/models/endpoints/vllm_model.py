import concurrent
import diskcache
import lighteval
import hashlib
import logging
import requests
import typing
import rich
import json

logger = logging.getLogger(__name__)


class VLLMClientConfig(lighteval.models.abstract_model.ModelConfig):
    model_name: str 
    base_url: str
    api_key: str = ""
    max_parallel: int = 10
    timeout: int = 600
    max_tokens: int|None = None
    max_parallel: int|None = 10
    extra_body: dict = {}
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int|None = None
    cache_dir: str|None

class VLLMClient(lighteval.models.abstract_model.LightevalModel):

    def __init__(self, config: VLLMClientConfig = None) -> None:
        self.config = config
        self._cache = lighteval.utils.cache_management.SampleCache(config.model_copy(update={"cache_dir": "/tmp"}))

    #@lighteval.utils.cache_management.cached(lighteval.tasks.requests.SamplingMethod.GENERATIVE)
    def greedy_until(self, docs: list[lighteval.tasks.requests.Doc]) -> list[lighteval.models.model_output.ModelResponse]:

        def call(doc: lighteval.tasks.requests.Doc) -> lighteval.models.model_output.ModelResponse:
            """ Make a single request to the model endpoint and return the response. """

            # prepare request
            request_params = {
                "url" : f"{self.config.base_url}/chat/completions",
                "headers": {
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                "json" : {
                    "model": self.config.model_name,
                    "messages": (
                        messages := (
                            [{"role":"system", "content": doc.instruction}] if doc.instruction else []) + 
                            [{"role": "user", "content": doc.query}]
                        ),
                    "n": doc.num_samples,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "min_p": self.config.min_p,
                    "frequency_penalty": self.config.frequency_penalty,
                    "presence_penalty": self.config.presence_penalty,
                    "seed": self.config.seed,
                    **self.config.extra_body,
                },
                "timeout" : self.config.timeout,
            }

            # check cache
            with diskcache.Cache(self.config.cache_dir or "/tmp/vllm_cache") as cache:
                key = hashlib.sha256(json.dumps(request_params, sort_keys=True).encode()).hexdigest()
                if key not in cache: cache[key] = {"response" : requests.post(**request_params).json(), "request" : request_params}
                response = cache[key]["response"]

            # check for errors
            if response.get("code",200) != 200:
                raise RuntimeError(f"VLLM API error: {response}")

            # parse response
            response = lighteval.models.model_output.ModelResponse(
                text=[choice["message"]["content"] for choice in response["choices"]],
                reasonings=[choice["message"]["reasoning_content"] for choice in response["choices"]],
                input=messages,
            )

            return response

        responses: typing.List[typing.Optional[lighteval.models.model_output.ModelResponse]] = [None] * len(docs)
    
        with rich.progress.Progress("[progress.description]{task.description}", rich.progress.BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", "•", rich.progress.TimeElapsedColumn(), "•", rich.progress.TimeRemainingColumn()) as pbar:
            pbar.add_task(description="Sending requests...", total=len(docs))
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_parallel) as executor:
                futures = {executor.submit(call, doc): idx for idx, doc in enumerate(docs)}

                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    responses[idx] = future.result()
                    pbar.update(0, advance=1)

        return responses

    def cleanup(self):
        pass

    @property
    def tokenizer(self):
        raise NotImplementedError("VLLM client does not have a tokenizer.")

    @property
    def add_special_tokens(self) -> bool:
        raise NotImplementedError("VLLM client does not support special tokens.")

    @property
    def max_length(self) -> int:
        raise NotImplementedError("VLLM client does not have a max length.")

    #@lighteval.utils.cache_management.cached(lighteval.tasks.requests.SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[lighteval.tasks.requests.Doc]) -> list[lighteval.models.model_output.ModelResponse]:
        raise NotImplementedError("VLLM client does not support loglikelihood computation.")

    #@lighteval.utils.cache_management.cached(lighteval.tasks.requests.SamplingMethod.PERPLEXITY)
    def loglikelihood_rolling(self, docs: list[lighteval.tasks.requests.Doc]) -> list[lighteval.models.model_output.ModelResponse]:
        raise NotImplementedError("VLLM client does not support rolling loglikelihood computation.")
