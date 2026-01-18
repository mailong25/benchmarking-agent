from pydantic import BaseModel
from typing import List, Dict, Literal

class ComplianceInfo(BaseModel):
    analysis: str
    summary: str    

class EligibilityInfo(BaseModel):
    reasoning: str
    advisable: str
    sources: List[str]

class BenchmarkItem(BaseModel):
    name: str
    description: str

class DomainBenchmarks(BaseModel):
    domain: str
    datasets: List[BenchmarkItem]

class BenchmarkInfo(BaseModel):
    benchmarks: List[DomainBenchmarks]

class HFDatasetConfig(BaseModel):
    path: str
    name: str
    split: str

class TaskPromptResponse(BaseModel):
    prompt: str

class FunctionResponse(BaseModel):
    reasoning: str
    function: str | None

class LiteLLMModelInfo(BaseModel):
    litellm_models: List[str]

class LLMasJudgeMetric(BaseModel):
    metric_name: str
    metric_score_min: float | int
    metric_score_max: float | int
    prompt: str

class MetricFunction(BaseModel):
    metric_name: str
    metric_score_min: float | int
    metric_score_max: float | int
    function: str

class SimpleResponse(BaseModel):
    answer: str

class ScoringResponse(BaseModel):
    reasoning: str
    score: int | float

class TrustworthyCode(BaseModel):
    reasoning: str
    trustworthy: bool

class EvalGuide(BaseModel):
    input_output_reasoning: str
    input_fields: List[str]
    reference_field: str | None
    output_nature: str
    task_category: str
    evaluation_type_reasoning: str
    evaluation_method_reasoning: str
    use_llm_as_judge: bool

class MetaEvaluation(BaseModel):
    verdict: Literal["failed", "passed", "inconclusive"]
    failed_step: Literal["eval_guide", "prompt_template", "evaluator"] | None
    suggestion: str | None