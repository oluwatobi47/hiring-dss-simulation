#!/usr/bin/env python3
"""Run the local Llama.cpp hiring DSS simulation from the notebook workflow."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional, Sequence, Union

import torch
from llama_index.core import Document, Response, Settings, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.token_counting import TokenCountingHandler
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms import LLM
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.readers.base import BaseReader
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    DEFAULT_SYSTEM_PROMPT,
    completion_to_prompt_v3_instruct,
    messages_to_prompt_v3_instruct,
)
from llama_index.readers.file import DocxReader, PDFReader
from pydantic import BaseModel, PrivateAttr


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "simulation"
DEFAULT_MODEL_BASE_PATH = REPO_ROOT / "model"

DEFAULT_MODEL_FILES = {
    "8b": "llama31_8b_hiring_fp16.gguf",
    "8b_Q4": "llama31_8b_hiring_Q4_K_M.gguf",
    "8b_Q8": "llama31_8b_hiring_Q8_0.gguf",
}

DEFAULT_LLAMA_CPP_MODEL_VERBOSITY = True


def add_to_json_file(json_file: Path, new_data: dict[str, Any]) -> None:
    data: list[Any] = []
    try:
        with json_file.open("r", encoding="utf-8") as file:
            try:
                data = json.load(file)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
            data.append(new_data)
    except FileNotFoundError:
        data = [new_data]

    json_file.parent.mkdir(parents=True, exist_ok=True)
    with json_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def read_json(json_file_path: Path) -> Union[dict[str, Any], list[Any]]:
    with json_file_path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def extract_structured_data(string_value: str, output_format: str = "json") -> str:
    del output_format
    json_patterns = (
        r"```json\s*({.*?})\s*```",
        r"```json\s*({.*?})\s*",
        r"```\s*({.*?})\s```*",
    )
    for pattern in json_patterns:
        match = re.search(pattern, string_value, re.DOTALL)
        if match:
            return match.group(1)
    return ""


def parse_json(string_value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(string_value)
        return parsed if isinstance(parsed, dict) else {}
    except json.decoder.JSONDecodeError:
        return {}


@dataclass
class Company:
    id: str
    name: str
    date_founded: str
    mission_statement: str
    vision: str
    company_culture_statement: str
    address: str


@dataclass
class JobDescription:
    id: str
    company_id: str
    job_title: str
    job_description_file: str


@dataclass
class JobPost:
    id: str
    created_date: str
    title: str
    job_description_id: str
    active: bool
    salary_range: str


@dataclass
class JobApplication:
    id: str
    candidate_name: str
    candidate_email: str
    created_date: str
    job_post_id: str
    resume_link: str
    active: bool


@dataclass
class DataPool:
    job_descriptions: list[JobDescription]
    job_posts: list[JobPost]
    job_applications: list[JobApplication]
    company_info: Optional[Company] = None


class RuntimeLLMArgs(BaseModel):
    system_prompt: str
    max_new_tokens: int = 2048
    verbose: bool = DEFAULT_LLAMA_CPP_MODEL_VERBOSITY
    temperature: float = DEFAULT_TEMPERATURE


class RuntimeLlamaLLM(LLM):
    _llm: LLM = PrivateAttr()
    generate_kwargs: dict[str, Any] = {}
    messages_to_prompt: Any = None
    completion_to_prompt: Any = None
    max_new_tokens: Any = None
    model_kwargs: dict[str, Any] = {}

    def __init__(self, llm: LLM, model_config: RuntimeLLMArgs):
        model_kwargs = {**(llm.model_kwargs or {}), **llm.model_dump()}
        model_kwargs.pop("class_name", None)
        model_kwargs["query_wrapper_prompt"] = llm.query_wrapper_prompt
        super().__init__(**model_kwargs)

        self.model_kwargs = model_kwargs
        self._llm = llm
        self.generate_kwargs = self._llm.generate_kwargs
        self.system_prompt = model_config.system_prompt

        def msg_to_prompt_fn(messages: Sequence[ChatMessage], system_prompt: str = self.system_prompt):
            return messages_to_prompt_v3_instruct(messages, system_prompt=system_prompt)

        def comp_to_prompt_fn(prompt: str, system_prompt: str = self.system_prompt):
            return completion_to_prompt_v3_instruct(prompt, system_prompt=system_prompt)

        self.messages_to_prompt = msg_to_prompt_fn
        self.completion_to_prompt = comp_to_prompt_fn
        self.max_new_tokens = model_config.max_new_tokens
        self.generate_kwargs.update({"temperature": model_config.temperature})

    @classmethod
    def class_name(cls) -> str:
        return "LlamaCPP_custom_runtime_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return self._llm.metadata

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self._llm.generate_kwargs.update(self.generate_kwargs)
        if not formatted:
            prompt = self.completion_to_prompt(prompt)
        return self._llm.complete(prompt, True, **kwargs)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        self.generate_kwargs.update({"stream": True})
        self._llm.generate_kwargs.update(self.generate_kwargs)
        if not formatted:
            prompt = self.completion_to_prompt(prompt)
        return self._llm.stream_complete(prompt, True, **kwargs)

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **(kwargs or {}))

    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, formatted=formatted, **(kwargs or {}))

    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        raise ValueError("Not implemented")

    @llm_completion_callback()
    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        raise ValueError("Not implemented")


def resolve_dataset_path(base_dataset_folder: Path, dataset_ref: str) -> Path:
    return base_dataset_folder / dataset_ref.lstrip("/")


def process_file_to_docs(file_path: Path, metadata: dict[str, str], text_data: str | None = None):
    file_readers: dict[str, BaseReader] = {
        ".pdf": PDFReader(return_full_document=True),
        ".docx": DocxReader(),
    }
    reader = file_readers.get(file_path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported file type: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"Referenced dataset file does not exist: {file_path}")

    docs = reader.load_data(file_path)
    if docs:
        if text_data:
            docs[0] = Document(text=f"{text_data}\n{docs[0].text}", metadata=docs[0].metadata)

        for doc in docs:
            doc.metadata.update(metadata)
            doc.excluded_llm_metadata_keys.extend(["file_name", "file_path", "file_type", "file_size"])
            doc.excluded_embed_metadata_keys.extend(["file_name", "file_path", "file_type", "file_size"])
    return docs


class DataIngestionService:
    def __init__(self, base_dataset_folder: Path, embedding_model: HuggingFaceEmbedding, score_data: dict[str, Any]):
        self._embedding_model = embedding_model
        self._score_data = score_data
        self._base_dataset_folder = base_dataset_folder
        self._doc_store: dict[str, list[Document]] = {}
        self._vector_index_map: dict[str, VectorStoreIndex] = {}
        self._criteria_id_map = {
            "1": "Experience",
            "2": "Skills",
            "3": "Academic Qualifications",
            "4": "Certifications",
            "5": "Soft Skills",
        }

    def index_job_application(self, job_application: JobApplication) -> None:
        data_text = f"""
        Job Application Data for Application ID: {job_application.id}

        Candidate Name: {job_application.candidate_name}
        Candidate Email: {job_application.candidate_email}
        Application Date: {job_application.created_date}
        Job Post ID: {job_application.job_post_id}
        """

        score_output = ["Application score data:"]
        for score_entry in self._score_data.get(job_application.id, []):
            criterion = self._criteria_id_map.get(str(score_entry.get("criterion_id")))
            if criterion is not None:
                score_output.append(
                    f"  Criterion: {criterion}\n"
                    f"  Score: {score_entry.get('score', 'N/A')} / {score_entry.get('max_score', 'N/A')}\n"
                    f"  Explanation: {score_entry.get('explanation', 'No explanation provided.')}\n"
                    f"  Assessment Feedback: {score_entry.get('feedback', 'No feedback provided.')}\n"
                )

        if len(score_output) > 1:
            data_text += "\n" + "\n".join(score_output)

        docs = process_file_to_docs(
            file_path=resolve_dataset_path(self._base_dataset_folder, job_application.resume_link),
            text_data=data_text,
            metadata={
                "job_application_reference": job_application.id,
                "job_post_reference": job_application.job_post_id,
            },
        )
        docs.append(
            Document(
                text=data_text,
                metadata={
                    "job_application_reference": job_application.id,
                    "job_post_reference": job_application.job_post_id,
                },
            )
        )
        self._doc_store.setdefault("job_applications", []).extend(docs)

    def index_job_description(self, job_description: JobDescription) -> None:
        data_text = f"""
        Job Description Data with ID: {job_description.id}

        Job Description Name: {job_description.job_title}
        """
        docs = process_file_to_docs(
            file_path=resolve_dataset_path(self._base_dataset_folder, job_description.job_description_file),
            metadata={"job_description_reference": job_description.id},
            text_data=data_text,
        )
        self._doc_store.setdefault("job_descriptions", []).extend(docs)

    def index_job_post(self, job_post: JobPost) -> None:
        data_text = f"""
        Job Post Data with ID: {job_post.id}

        Job Description ID: {job_post.job_description_id}
        Job Post Title: {job_post.title}
        Salary Range: {job_post.salary_range}
        Posted Date: {job_post.created_date}
        """
        docs = [
            Document(
                text=data_text,
                metadata={
                    "job_post_reference": job_post.id,
                    "job_description_reference": job_post.job_description_id,
                },
            )
        ]
        self._doc_store.setdefault("job_posts", []).extend(docs)

    def index_company_info(self, company: Company) -> None:
        data_text = f"""
        Company Information: {company.id}

        Company Name: {company.name}
        Company Vision: {company.vision}
        Company Mission: {company.mission_statement}
        Company Culture Statement: {company.company_culture_statement}
        Address: {company.address}
        Date Founded: {company.date_founded}
        """
        self._doc_store.setdefault("company_info", []).append(
            Document(text=data_text, metadata={"parent_obj_ref": company.id})
        )

    def write_data_to_stores(self, datapool: DataPool) -> None:
        for application in datapool.job_applications:
            self.index_job_application(application)
        for description in datapool.job_descriptions:
            self.index_job_description(description)
        for post in datapool.job_posts:
            self.index_job_post(post)
        if datapool.company_info is not None:
            self.index_company_info(datapool.company_info)

    def get_store_index(
        self,
        store: Literal["job_posts", "job_descriptions", "job_applications", "company_info"],
    ) -> VectorStoreIndex:
        if store not in self._vector_index_map:
            documents = self._doc_store[store]
            self._vector_index_map[store] = VectorStoreIndex.from_documents(
                documents=documents,
                embed_model=self._embedding_model,
            )
        return self._vector_index_map[store]


class ToolOutput(BaseModel):
    tool: str
    output: str
    context: Any


class ToolQuery(BaseModel):
    tool: str
    query: str


class ContextRetrievalEvent(Event):
    tool: Optional[str] = None
    context: Optional[str] = None
    query: str


class JobPostContextEvent(ContextRetrievalEvent):
    tool: str = "job_post_tool"


class JobApplicationContextEvent(ContextRetrievalEvent):
    tool: str = "job_application_tool"


class JobDescriptionContextEvent(ContextRetrievalEvent):
    tool: str = "job_description_tool"


class CompanyInfoContextEvent(ContextRetrievalEvent):
    tool: str = "company_information_tool"


class ContextOutputEvent(Event):
    tool: str
    query: str
    output: str


class ContextProcessOutputEvent(Event):
    user_query: str
    context: list[str]
    tool_path: list[str]


class ContextAgentResult(BaseModel):
    tool_path: list[str]
    context: list[str]


class QuestionBreakDownEvent(Event):
    user_question: str
    questions: list[str]


class ContextBuilderAgentWorkflow(Workflow):
    def __init__(
        self,
        job_application_query_engine: BaseQueryEngine,
        job_post_query_engine: BaseQueryEngine,
        job_description_query_engine: BaseQueryEngine,
        company_info_query_engine: BaseQueryEngine,
        llm: LLM,
        max_tool_use: int = 1,
    ):
        super().__init__(timeout=300)
        self.max_tool_use = max_tool_use
        self.job_application_query_engine = job_application_query_engine
        self.job_post_query_engine = job_post_query_engine
        self.job_description_query_engine = job_description_query_engine
        self.company_info_query_engine = company_info_query_engine
        self.tools = ["job_application_tool", "job_post_tool", "job_description_tool", "company_information_tool"]
        self.llm = llm

    async def update_tool_use_context(self, ctx: Context, tool: str) -> None:
        tool_use_data = await ctx.get("tool_use_tracker", {})
        tool_use_data[tool] = (tool_use_data[tool] if tool in tool_use_data and tool_use_data[tool] >= 0 else 0) + 1
        await ctx.set("tool_use_tracker", tool_use_data)

    async def tool_use_limit_reached(self, ctx: Context, tool: str | None) -> bool:
        if not tool:
            return False
        tool_use_data = await ctx.get("tool_use_tracker", {})
        return tool_use_data[tool] >= self.max_tool_use if tool in tool_use_data and tool_use_data[tool] else False

    def process_query_engine_context_retrieval(
        self,
        engine: BaseQueryEngine,
        event: ContextRetrievalEvent,
    ) -> ContextOutputEvent:
        prompt = f"""
        ## Question:
        {event.query}
        """
        if event.context:
            prompt += f"""
            ## Context:
            {event.context}
            """
        response: Response = engine.query(prompt)
        return ContextOutputEvent(tool=event.tool, output=response.response.strip(), query=event.query)

    @step
    async def router_agent_node(
        self,
        ctx: Context,
        ev: StartEvent,
    ) -> JobApplicationContextEvent | JobPostContextEvent | JobDescriptionContextEvent | CompanyInfoContextEvent:
        answer_relevancy_prompt = f"""
        Given a query and a list of tools, choose the best tool that can provide relevant context.

        Tools:
        job_application_tool: Provides information about candidates and job applications
        job_post_tool: Provides information on job posts that candidates apply to and salary details for such positions
        job_description_tool: Provides detailed information about jobs and job details
        company_information_tool: Provides information about a company the candidate is applying to

        Return only one tool name and no additional text.

        Query:
        {ev.query}

        Response:
        """
        selection_response = self.llm.complete(answer_relevancy_prompt)
        tool = selection_response.text.strip()
        limit_reached = await self.tool_use_limit_reached(ctx, tool)

        if tool in self.tools and not limit_reached:
            if tool == "job_application_tool":
                return JobApplicationContextEvent(query=ev.query)
            if tool == "job_post_tool":
                return JobPostContextEvent(query=ev.query)
            if tool == "job_description_tool":
                return JobDescriptionContextEvent(query=ev.query)
            if tool == "company_information_tool":
                return CompanyInfoContextEvent(query=ev.query)
        return JobApplicationContextEvent(query=ev.query)

    @step
    async def job_application_agent_node(self, event: JobApplicationContextEvent) -> ContextOutputEvent:
        return self.process_query_engine_context_retrieval(self.job_application_query_engine, event)

    @step
    async def job_post_agent_node(self, event: JobPostContextEvent) -> ContextOutputEvent:
        return self.process_query_engine_context_retrieval(self.job_post_query_engine, event)

    @step
    async def job_description_agent_node(self, event: JobDescriptionContextEvent) -> ContextOutputEvent:
        return self.process_query_engine_context_retrieval(self.job_description_query_engine, event)

    @step
    async def company_info_agent_node(self, event: CompanyInfoContextEvent) -> ContextOutputEvent:
        return self.process_query_engine_context_retrieval(self.company_info_query_engine, event)

    @step
    async def context_evaluator_agent_node(
        self,
        ctx: Context,
        ev: ContextOutputEvent,
    ) -> JobApplicationContextEvent | JobPostContextEvent | JobDescriptionContextEvent | CompanyInfoContextEvent | StopEvent:
        await self.update_tool_use_context(ctx, ev.tool)
        tool_paths = await ctx.get("tool_path", [])
        context = await ctx.get("generated_context", [])
        context.append(ev.output)
        tool_paths.append(ev.tool)
        await ctx.set("tool_path", tool_paths)
        await ctx.set("generated_context", context)

        context_string = "\n".join(context)
        context_tools = "\n".join([f"{index + 1}. {tool}" for index, tool in enumerate(tool_paths)])
        answer_relevancy_prompt = f"""
        Determine if the current context is sufficient to answer the query. If not, choose the next best tool.

        Defined Tools:
        job_application_tool: Provides information about candidates and job applications
        job_post_tool: Provides information on job posts that candidates apply to and salary details for such positions
        job_description_tool: Contains job description information
        company_information_tool: Provides information about a company the candidate is applying to

        Tool context relationships:
        - job_application references job_post
        - job_post references job_description
        - job_description references company_information

        Return only this JSON inside markdown JSON delimiters:
        ```json
        {{
            "sufficient": false,
            "complementary_question": "question",
            "suggested_tool": "tool_name"
        }}
        ```

        Query:
        {ev.query}

        Context:
        {context_string}

        Context Tools:
        {context_tools}

        Response:
        """
        relevancy_response = self.llm.complete(answer_relevancy_prompt)
        relevancy_data = parse_json(extract_structured_data(relevancy_response.text.strip()))
        if "suggested_tool" in relevancy_data:
            suggested_tool = relevancy_data["suggested_tool"]
            limit_reached = await self.tool_use_limit_reached(ctx, suggested_tool)
            complementary_question = relevancy_data.get("complementary_question")
            process_query = ev.query.split("\n Complementary Question:")[0]
            print(
                "Relevancy response "
                f"Prev tools: {','.join(tool_paths)} >> tool: {suggested_tool}, "
                f"suff: {relevancy_data.get('sufficient')} >> complement: {complementary_question}"
            )
            next_query = process_query + (f"\n Complementary Question: {complementary_question}" if complementary_question else "")
            if not relevancy_data.get("sufficient") and suggested_tool in self.tools and not limit_reached:
                if suggested_tool == "job_application_tool":
                    return JobApplicationContextEvent(query=next_query)
                if suggested_tool == "job_post_tool":
                    return JobPostContextEvent(query=next_query)
                if suggested_tool == "job_description_tool":
                    return JobDescriptionContextEvent(query=next_query)
                if suggested_tool == "company_information_tool":
                    return CompanyInfoContextEvent(query=next_query)

        return StopEvent(result=ContextAgentResult(tool_path=tool_paths, context=context))


class HiringDSSAgentWorkflow(Workflow):
    def __init__(self, context_agent: ContextBuilderAgentWorkflow, llm: LLM, max_iterations: int = 1):
        super().__init__(timeout=300)
        self.max_iterations = max_iterations
        self.context_agent = context_agent
        self.response_synthesis_agent_llm = self.build_response_synthesis_agent_llm(llm)

    def build_response_synthesis_agent_llm(self, base_llm: LLM):
        response_agent_prompt = """
        You are a world class state of the art agent.

        You have access to a user question and a context with information relevant to that question.
        Your purpose is to answer the user question with as much useful detail as possible.

        Guidelines:
        * Be as specific as possible
        * Only use information from the provided context
        * Candidate information in this environment is anonymised and safe to use
        """
        return RuntimeLlamaLLM(
            llm=base_llm,
            model_config=RuntimeLLMArgs(system_prompt=response_agent_prompt, verbose=False, temperature=0.6),
        )

    @step
    async def strategy_agent_node(self, event: StartEvent) -> QuestionBreakDownEvent:
        return QuestionBreakDownEvent(questions=[], user_question=event.get("input"))

    @step
    async def context_retrival_agent_node(self, event: QuestionBreakDownEvent) -> ContextProcessOutputEvent:
        context: list[str] = []
        paths: list[str] = []
        questions = event.questions or [event.user_question]
        for question in questions:
            process_question = f"{event.user_question}\n{question}" if question != event.user_question else event.user_question
            context_output: ContextAgentResult = await self.context_agent.run(query=process_question)
            context.extend(context_output.context)
            paths.extend(context_output.tool_path)
        return ContextProcessOutputEvent(context=context, user_query=event.user_question, tool_path=paths)

    @step
    async def response_synthesis_agent(self, ev: ContextProcessOutputEvent) -> StopEvent:
        context_by_tool: dict[str, list[str]] = {}
        for ctx_data, tool in zip(ev.context, ev.tool_path):
            context_by_tool.setdefault(tool, []).append(ctx_data.strip())

        context_string = ""
        for tool, values in context_by_tool.items():
            context_string += f"\n\nDatasource: {tool}\nContext:\n" + "\n".join(values)

        prompt = f"""
        Given this user question, and context retrieved from different data sources based on the question,
        generate a response to the user question. Return only the response.

        ## User Question:
        {ev.user_query}

        ## Context:
        {context_string}

        ## Response:
        """
        response = self.response_synthesis_agent_llm.complete(prompt)
        return StopEvent(
            result={
                "response": {"text": response.text},
                "context": ev.context,
                "paths": ev.tool_path,
            }
        )


def get_test_cases(test_case_config: dict[str, Any]) -> list[dict[str, Any]]:
    return list(test_case_config.values())


def get_simulation_data(test_case_code: str, datapool: dict[str, Any], test_case_config: dict[str, Any]) -> DataPool:
    full_pool = DataPool(
        company_info=Company(**datapool["company_info"]),
        job_descriptions=[JobDescription(**item) for item in datapool["job_descriptions"]],
        job_posts=[JobPost(**item) for item in datapool["job_posts"]],
        job_applications=[JobApplication(**item) for item in datapool["job_applications"]],
    )
    config = test_case_config[test_case_code]
    return DataPool(
        company_info=full_pool.company_info,
        job_descriptions=full_pool.job_descriptions[: config["job_descriptions"]],
        job_posts=full_pool.job_posts[: config["job_posts"]],
        job_applications=full_pool.job_applications[: config["job_applications"]],
    )


async def run_simulation(
    test_case: str,
    questions: list[dict[str, Any]],
    run_function: Callable[..., Awaitable[Any]],
    output_path: Path,
    token_counter: TokenCountingHandler,
    progress_state: dict[str, Any],
) -> None:
    def render_progress(done: int, total: int, width: int = 30) -> str:
        if total <= 0:
            return "[" + ("-" * width) + "]   0.0%"
        ratio = done / total
        filled = min(width, max(0, int(ratio * width)))
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {ratio * 100:5.1f}%"

    for index, question in enumerate(questions):
        error = None
        response = None
        start_time = time.perf_counter()
        try:
            print(f"[{test_case}] Question {index + 1}/{len(questions)}: {question['question']}")
            response = await run_function(input=question["question"])
        except Exception as exc:
            traceback.print_exc()
            error = str(exc)
        finally:
            end_time = time.perf_counter()
            prompt_token = token_counter.prompt_llm_token_count
            response_token = token_counter.completion_llm_token_count
            token_counter.reset_counts()

        response_data = response["response"] if response else None
        metric_data = {
            "batch_id": test_case,
            "question_id": question["number"],
            "prompt": question["question"],
            "expected_output": question["response"],
            "model_response": response_data["text"] if response_data else None,
            "context": response["context"] if response else [],
            "error": error,
            "processing_time": end_time - start_time,
            "processing_path": response["paths"] if response else [],
            "power_consumption": [],
            "prompt_token_count": prompt_token,
            "response_token_count": response_token,
        }
        add_to_json_file(output_path / f"TC_OUTPUT_{test_case}.json", metric_data)

        progress_state["completed"] += 1
        completed = progress_state["completed"]
        total = progress_state["total"]
        elapsed = time.perf_counter() - progress_state["started_at"]
        eta_seconds = (elapsed / completed) * (total - completed) if completed > 0 else 0.0
        print(
            f"Progress {render_progress(completed, total)} "
            f"({completed}/{total}) | elapsed {elapsed:.1f}s | ETA {eta_seconds:.1f}s"
        )


def get_model_args(prompt: str, temperature: float = 0.6) -> RuntimeLLMArgs:
    return RuntimeLLMArgs(system_prompt=prompt, verbose=False, temperature=temperature)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Llama.cpp local hiring DSS simulation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Folder containing simulation JSON files and dataset files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Folder for output JSON files. Defaults to <data-dir>/output/gguf/workflow_<model>.")
    parser.add_argument("--model-base-path", type=Path, default=DEFAULT_MODEL_BASE_PATH, help="Folder containing GGUF model files and embedding cache.")
    parser.add_argument("--model-path", type=Path, default=None, help="Direct path to a GGUF model. Overrides --simulation-model.")
    parser.add_argument("--simulation-model", choices=sorted(DEFAULT_MODEL_FILES), default="8b_Q4", help="Named model file under --model-base-path.")
    parser.add_argument("--embedding-model-name", default="BAAI/bge-m3", help="Hugging Face embedding model name or local path.")
    parser.add_argument("--embedding-cache-dir", type=Path, default=None, help="Embedding cache folder. Defaults to <model-base-path>/embedding.")
    parser.add_argument("--test-case", action="append", help="Run one test case code, e.g. TC1. Repeat to run multiple. Defaults to all.")
    parser.add_argument("--question-limit", type=int, default=None, help="Only run the first N questions from truthful_qa_questions.json.")
    parser.add_argument("--similarity-top-k", type=int, default=2)
    parser.add_argument("--max-tool-use", type=int, default=1)
    parser.add_argument("--context-window", type=int, default=16000)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--retrieval-temperature", type=float, default=0.3)
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="llama.cpp GPU layers. Defaults to -1 (all layers). Use 0 for CPU-only.")
    parser.add_argument("--verbose-llama", action="store_true", help="Enable llama.cpp verbose logging.")
    parser.add_argument("--skip-cuda-cleanup", action="store_true", help="Skip torch CUDA cache cleanup at exit.")
    return parser


def validate_inputs(data_dir: Path, model_path: Path) -> None:
    required_json = [
        "application_score_data.json",
        "test_case_config.json",
        "truthful_qa_questions.json",
        "evaluation_data_pool.json",
    ]
    missing = [str(data_dir / name) for name in required_json if not (data_dir / name).exists()]
    if not model_path.exists():
        missing.append(str(model_path))
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(f"- {path}" for path in missing))


async def main_async(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    model_base_path = args.model_base_path.resolve()
    model_path = args.model_path.resolve() if args.model_path else model_base_path / DEFAULT_MODEL_FILES[args.simulation_model]
    output_dir = args.output_dir or data_dir / "output" / "gguf" / f"workflow_{args.simulation_model}"
    embedding_cache_dir = args.embedding_cache_dir or model_base_path / "embedding"

    validate_inputs(data_dir, model_path)

    score_data = read_json(data_dir / "application_score_data.json")
    test_case_config = read_json(data_dir / "test_case_config.json")
    truthful_qa_questions = read_json(data_dir / "truthful_qa_questions.json")
    datapool = read_json(data_dir / "evaluation_data_pool.json")

    if not isinstance(score_data, dict) or not isinstance(test_case_config, dict) or not isinstance(datapool, dict):
        raise ValueError("Simulation JSON files are not in the expected object format.")
    if not isinstance(truthful_qa_questions, list):
        raise ValueError("truthful_qa_questions.json must contain a list of questions.")

    selected_cases = args.test_case or [tc["code"] for tc in get_test_cases(test_case_config)]
    unknown_cases = [tc for tc in selected_cases if tc not in test_case_config]
    if unknown_cases:
        raise ValueError(f"Unknown test case(s): {', '.join(unknown_cases)}")
    questions = truthful_qa_questions[: args.question_limit] if args.question_limit else truthful_qa_questions

    print(f"Loading embedding model: {args.embedding_model_name}")
    embedding_model = HuggingFaceEmbedding(
        model_name=args.embedding_model_name,
        cache_folder=str(embedding_cache_dir),
    )

    print(f"Loading GGUF model: {model_path}")
    base_llm = LlamaCPP(
        model_path=str(model_path),
        context_window=args.context_window,
        model_kwargs={
            "chat_format": "llama-3",
            "f16_kv": True,
            "n_gpu_layers": args.n_gpu_layers,
        },
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose_llama,
        messages_to_prompt=messages_to_prompt_v3_instruct,
        completion_to_prompt=completion_to_prompt_v3_instruct,
    )

    def tokenize(text: str):
        return base_llm._model.tokenize(text.encode("utf-8"), False, False)

    token_counter = TokenCountingHandler(tokenizer=tokenize)
    Settings.callback_manager = CallbackManager([token_counter])

    running_llm = RuntimeLlamaLLM(llm=base_llm, model_config=get_model_args(DEFAULT_SYSTEM_PROMPT, args.temperature))
    ja_ret_llm = RuntimeLlamaLLM(llm=base_llm, model_config=get_model_args(JOB_APPLICATION_TOOL_SYSTEM_PROMPT, args.retrieval_temperature))
    jd_ret_llm = RuntimeLlamaLLM(llm=base_llm, model_config=get_model_args(JOB_DESCRIPTION_TOOL_SYSTEM_PROMPT, args.retrieval_temperature))
    jp_ret_llm = RuntimeLlamaLLM(llm=base_llm, model_config=get_model_args(JOB_POST_TOOL_SYSTEM_PROMPT, args.retrieval_temperature))
    c_ret_llm = RuntimeLlamaLLM(llm=base_llm, model_config=get_model_args(COMPANY_INFORMATION_TOOL_SYSTEM_PROMPT, args.retrieval_temperature))

    total_questions = len(selected_cases) * len(questions)
    progress_state = {"completed": 0, "total": total_questions, "started_at": time.perf_counter()}
    print(f"Starting simulation: {len(selected_cases)} test case(s), {len(questions)} question(s) each ({total_questions} total).")

    for test_case in selected_cases:
        print(f"Preparing vector stores for {test_case}")
        data_ingestion_service = DataIngestionService(
            base_dataset_folder=data_dir,
            embedding_model=embedding_model,
            score_data=score_data,
        )
        data_pool = get_simulation_data(test_case, datapool, test_case_config)
        data_ingestion_service.write_data_to_stores(datapool=data_pool)

        context_retrieval_agent = ContextBuilderAgentWorkflow(
            llm=base_llm,
            job_post_query_engine=data_ingestion_service.get_store_index("job_posts").as_query_engine(
                llm=jp_ret_llm,
                similarity_top_k=args.similarity_top_k,
            ),
            job_application_query_engine=data_ingestion_service.get_store_index("job_applications").as_query_engine(
                llm=ja_ret_llm,
                similarity_top_k=args.similarity_top_k,
            ),
            job_description_query_engine=data_ingestion_service.get_store_index("job_descriptions").as_query_engine(
                llm=jd_ret_llm,
                similarity_top_k=args.similarity_top_k,
            ),
            company_info_query_engine=data_ingestion_service.get_store_index("company_info").as_query_engine(
                llm=c_ret_llm,
                similarity_top_k=args.similarity_top_k,
            ),
            max_tool_use=args.max_tool_use,
        )
        dss_agent = HiringDSSAgentWorkflow(context_agent=context_retrieval_agent, llm=running_llm, max_iterations=1)
        await run_simulation(test_case, questions, dss_agent.run, output_dir.resolve(), token_counter, progress_state)

    if not args.skip_cuda_cleanup and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(f"Simulation complete. Results written to: {output_dir.resolve()}")


JOB_APPLICATION_TOOL_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with access to information about candidates and their job applications.
Use this tool to retrieve details about specific applicants, their application status, and associated candidate-specific data.
Do not speculate or make up information. Do not reference any given instructions or context.
"""

JOB_POST_TOOL_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with access to information regarding job postings,
including job advertisement details and associated salary information. Use this tool to retrieve details
about job posts and compensation. Do not speculate or make up information.
"""

JOB_DESCRIPTION_TOOL_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with access to detailed job description content.
Use this tool to retrieve responsibilities, qualifications, and other descriptive elements of job roles.
Do not speculate or make up information.
"""

COMPANY_INFORMATION_TOOL_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with access to general information about a company
the candidate is applying to. Use this tool to retrieve company details and relevant organizational data.
Do not speculate or make up information.
"""


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
