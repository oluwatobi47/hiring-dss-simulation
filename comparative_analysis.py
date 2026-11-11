"""Offline, validated loading and paired summaries for the model comparison."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from model_comparison_registry import FAMILIES, comparison_manifest
METRICS = ["correct", "answer_relevant", "context_relevant"]
SCORES = ["relevancy_score", "context_relevancy_score"]


def manifest(root, experiment):
    return comparison_manifest(Path(root) / "data/simulation/output/comparison", experiment)


def records(folder, prefix, id_field):
    rows = []
    for path in sorted(Path(folder).glob(f"{prefix}*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"Expected list: {path}")
        for row in payload:
            if row.get(id_field) is None:
                raise ValueError(f"Missing {id_field}: {path}")
            rows.append(dict(row, test_case=path.stem.removeprefix(prefix), question_id=str(row[id_field])))
    frame = pd.DataFrame(rows)
    if not frame.empty and frame.duplicated(["test_case", "question_id"]).any():
        raise ValueError(f"Duplicate question IDs within test case: {folder}")
    return frame


def load_runs(runs):
    frames, audit = [], []
    for spec in runs:
        sim = records(spec["results"], "TC_OUTPUT_", "question_id")
        ev = records(spec["evaluation"], "TC_EVAL_", "id")
        keys = ["test_case", "question_id"]
        unmatched = 0
        if not ev.empty:
            simkeys = set(map(tuple, sim[keys].values)) if not sim.empty else set()
            unmatched = sum(tuple(x) not in simkeys for x in ev[keys].values)
        audit.append(dict(configuration=spec["configuration"], simulation_rows=len(sim),
                          evaluation_rows=len(ev), orphan_evaluations=unmatched,
                          manifest_present=(Path(spec["results"]) / "RUN_CONFIG.json").exists()))
        if sim.empty:
            continue
        if ev.empty:
            ev = pd.DataFrame(columns=keys + METRICS + ["correctness_reason"])
        for col in METRICS + SCORES + ["correctness_reason"]:
            if col not in ev:
                ev[col] = None
        frame = sim.merge(ev[keys + METRICS + SCORES + ["correctness_reason"]], on=keys, how="left", validate="one_to_one")
        reason = frame.correctness_reason.fillna("").astype(str)
        invalid = reason.str.contains("Judge response|Judge JSON", case=False, regex=True)
        for col in METRICS:
            frame[col] = frame[col].map(lambda v: float(v) if isinstance(v, bool) else np.nan)
            frame.loc[invalid, col] = np.nan
        frame["invalid_judge"] = invalid
        frame["scored"] = frame[METRICS].notna().all(axis=1)
        for col in SCORES:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame.loc[invalid | ~frame[col].between(0, 1), col] = np.nan
        frame["failed"] = frame.get("error", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str).ne("") | frame.get("model_response", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str).str.strip().eq("")
        for col in ["processing_time", "prompt_token_count", "response_token_count"]:
            frame[col] = pd.to_numeric(frame.get(col, np.nan), errors="coerce")
            frame.loc[frame[col] < 0, col] = np.nan
        frame["path_length"] = frame.get("processing_path", pd.Series(index=frame.index, dtype=object)).map(lambda x: len(x) if isinstance(x, list) else np.nan)
        for key in ["configuration", "family", "precision", "run"]:
            frame[key] = spec[key]
        frames.append(derive_metrics(frame))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), pd.DataFrame(audit)


def derive_metrics(frame):
    """Keep unknown judgments unknown; never rewrite raw judge decisions."""
    frame = frame.copy()
    valid = frame[METRICS].notna().all(axis=1)
    correct, answer, context = (frame[c].eq(1) for c in METRICS)
    flags = {
        "hallucination_proxy": answer & (~correct | ~context),
        "retrieval_failure_proxy": ~correct & ~context,
        "confident_hallucination_proxy": answer & ~correct & ~context,
        "judge_disagreement": correct & ~answer,
        "incorrect_answer": ~correct,
        "context_miss": ~context,
    }
    for name, values in flags.items():
        frame[name] = values.astype(float).where(valid)
    duration = frame.processing_time.where(frame.processing_time.gt(0))
    frame["output_throughput"] = frame.response_token_count / duration
    return frame


def summarize(frame, by=None):
    if frame.empty:
        return pd.DataFrame()
    by = by or ["configuration", "family", "precision"]
    grouped = frame.groupby(by, sort=False, dropna=True)
    result = grouped.agg(
        n=("question_id", "size"), scored=("scored", "sum"), accuracy=("correct", "mean"),
        correct_n=("correct", "count"), answer_relevance_n=("answer_relevant", "count"),
        context_relevance_n=("context_relevant", "count"),
        answer_relevance=("answer_relevant", "mean"), context_relevance=("context_relevant", "mean"),
        failure_rate=("failed", "mean"), mean_latency=("processing_time", "mean"),
        latency_n=("processing_time", "count"), median_latency=("processing_time", "median"),
        p95_latency=("processing_time", lambda x: x.quantile(.95)),
        prompt_tokens=("prompt_token_count", "mean"), response_tokens=("response_token_count", "mean"),
        mean_path_length=("path_length", "mean"), path_n=("path_length", "count"),
        output_throughput=("output_throughput", "mean"), throughput_n=("output_throughput", "count"),
        relevancy_score=("relevancy_score", "mean"), relevancy_score_n=("relevancy_score", "count"),
        context_relevancy_score=("context_relevancy_score", "mean"),
        context_relevancy_score_n=("context_relevancy_score", "count"),
        hallucination_proxy=("hallucination_proxy", "mean"),
        retrieval_failure_proxy=("retrieval_failure_proxy", "mean"),
        confident_hallucination_proxy=("confident_hallucination_proxy", "mean"),
        judge_disagreement_count=("judge_disagreement", "sum"),
        incorrect_answer_rate=("incorrect_answer", "mean"), context_miss_rate=("context_miss", "mean"))
    conditional = []
    for key, group in grouped:
        key = key if isinstance(key, tuple) else (key,)
        g = group.dropna(subset=METRICS)
        hit, miss, wrong = g.context_relevant.eq(1), g.context_relevant.eq(0), g.correct.eq(0)
        conditional.append(dict(zip(by, key),
            context_hit_n=int(hit.sum()), context_miss_n=int(miss.sum()), incorrect_n=int(wrong.sum()),
            failure_given_context_miss=g.loc[miss, "incorrect_answer"].mean(),
            success_given_context=g.loc[hit, "correct"].mean(),
            context_miss_share_among_errors=g.loc[wrong, "context_miss"].mean(),
            context_sensitivity=g.loc[hit, "correct"].mean() - g.loc[miss, "correct"].mean()))
    return result.reset_index().merge(pd.DataFrame(conditional), on=by, validate="one_to_one")


# Ordered (column, display label) pairs prevent the original axis/label swap.
PERFORMANCE_RADAR = [
    ("accuracy", "Correctness ↑"), ("answer_relevance", "Answer relevance ↑"),
    ("context_relevance", "Context relevance ↑"), ("hallucination_proxy", "Hallucination proxy ↓"),
    ("output_throughput_normalized", "Output throughput ↑"),
    ("mean_path_length_normalized", "Path entries ↓"), ("mean_latency_normalized", "Latency ↓")]
CONTEXT_RADAR = [
    ("failure_given_context_miss", "Failure given context miss ↓"),
    ("success_given_context", "Success given context ↑"),
    ("incorrect_answer_rate", "Incorrect answers ↓"),
    ("context_miss_share_among_errors", "Context miss among errors ↓"),
    ("context_miss_rate", "Context miss ↓")]


def radar_values(summary):
    result = summary.copy()
    for col in ["output_throughput", "mean_path_length", "mean_latency"]:
        maximum = result[col].max()
        # All-zero observations stay zero; an entirely unavailable axis stays missing.
        result[col + "_normalized"] = result[col] / maximum if maximum > 0 else result[col]
    return result


def bootstrap(values, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(2000)]
    return tuple(np.quantile(means, [.025, .975]))


def paired(frame, baseline="terra"):
    keys = ["run", "test_case", "question_id"]
    base = frame[frame.configuration.eq(baseline)]
    result = []
    for name, group in frame.groupby("configuration"):
        if name == baseline:
            continue
        joined = group.merge(base, on=keys, suffixes=("", "_baseline"), validate="one_to_one")
        for metric in ["correct", "processing_time"]:
            valid = joined[[metric, metric + "_baseline"]].dropna()
            delta = valid[metric] - valid[metric + "_baseline"]
            low, high = bootstrap(delta)
            result.append(dict(configuration=name, metric=metric, n=len(delta),
                               difference=delta.mean(), low=low, high=high))
    return pd.DataFrame(result)
