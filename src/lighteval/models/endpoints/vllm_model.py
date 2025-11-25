import concurrent
import diskcache
import lighteval
import requests
import hashlib
import logging
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
    system_prompt_template: str|None = "{instruction}"

class VLLMClient(lighteval.models.abstract_model.LightevalModel):

    def __init__(self, config: VLLMClientConfig = None) -> None:
        self.config = config
        self.model_name = config.model_name
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.max_parallel = config.max_parallel
        self.timeout = config.timeout
        self.max_tokens = config.max_tokens
        self.extra_body = config.extra_body
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.top_k = config.top_k
        self.min_p = config.min_p
        self.frequency_penalty = config.frequency_penalty
        self.presence_penalty = config.presence_penalty
        self.seed = config.seed
        self.cache_dir = config.cache_dir
        self.system_prompt_template = config.system_prompt_template

    def greedy_until(self, docs: list[lighteval.tasks.requests.Doc]) -> list[lighteval.models.model_output.ModelResponse]:

        def call(doc: lighteval.tasks.requests.Doc, cache:diskcache.Cache) -> lighteval.models.model_output.ModelResponse:
            """ Make a single request to the model endpoint and return the response. """

            system_prompt = self.system_prompt_template.format(instruction=doc.instruction if doc.instruction else "" )

            # prepare request
            json_params = { 
                "model": self.model_name,
                "messages": (
                    messages := (
                        [{"role":"system", "content": system_prompt}] if system_prompt else []) + 
                        [{"role": "user", "content": doc.query}]
                    ),
                "n": doc.num_samples,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "min_p": self.min_p,
                "top_k": self.top_k,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "seed": self.seed,
                **self.extra_body,
            }

            #import rich
            #rich.print(json_params)

            # check cache
            key = hashlib.sha256(json.dumps(json_params, sort_keys=True).encode()).hexdigest()
            if key in cache:
                response = cache[key]["response"]
                if response.get("code", 200) != 200:
                    logger.warning(f"VLLM API returned cached error response for key {key}: {response}, returning empty responses.")
            else:
                response = requests.post(
                    url=f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=json_params,
                    timeout=self.timeout
                ).json()
                cache[key] = {"response" : response, "request" : json_params}


            match response.get("code", 200):
                case 200:
                    return lighteval.models.model_output.ModelResponse(
                        text=[choice["message"]["content"] for choice in response["choices"]],
                        reasonings=[choice["message"]["reasoning_content"] for choice in response["choices"]],
                        input=messages,
                    )
                case 400:
                    logger.warning(f"VLLM API returned 400 Bad Request: {response}, returning empty responses.")
                    return lighteval.models.model_output.ModelResponse(
                        text=["" for _ in range(doc.num_samples)],
                        reasonings=["" for _ in range(doc.num_samples)],
                        input=messages,
                    )
                case _:
                    raise RuntimeError(f"VLLM API error: {response}")

        responses: typing.List[typing.Optional[lighteval.models.model_output.ModelResponse]] = [None] * len(docs)
    
        with rich.progress.Progress(
            "[progress.description]{task.description}", 
            rich.progress.BarColumn(), 
            "[progress.completed]{task.completed}/{task.total}", "•",
            rich.progress.TimeElapsedColumn(), "•", 
            rich.progress.TimeRemainingColumn()
        ) as pbar:
            with diskcache.Cache(self.cache_dir or "/tmp/vllm_cache") as cache:
                task = pbar.add_task(description="Sending requests...", total=len(docs))
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
                    futures = {executor.submit(call, doc, cache): idx for idx, doc in enumerate(docs)}

                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        responses[idx] = future.result()
                        pbar.update(task, advance=1)

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

    def loglikelihood(self, docs: list[lighteval.tasks.requests.Doc]) -> list[lighteval.models.model_output.ModelResponse]:
        raise NotImplementedError("VLLM client does not support loglikelihood computation.")

    def loglikelihood_rolling(self, docs: list[lighteval.tasks.requests.Doc]) -> list[lighteval.models.model_output.ModelResponse]:
        raise NotImplementedError("VLLM client does not support rolling loglikelihood computation.")
