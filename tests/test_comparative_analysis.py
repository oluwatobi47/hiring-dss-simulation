import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from comparative_analysis import (load_runs, manifest, paired, summarize, derive_metrics,
                                  radar_values, PERFORMANCE_RADAR)


class ComparisonTests(unittest.TestCase):
    def test_proxy_conditional_and_throughput_definitions(self):
        frame = pd.DataFrame({
            'correct': [1., 0., 1., 0., np.nan],
            'answer_relevant': [0., 1., 1., 0., np.nan],
            'context_relevant': [1., 0., 0., 1., np.nan],
            'processing_time': [2., 0., 4., 2., 2.],
            'response_token_count': [10, 10, 8, 4, 4],
        })
        derived = derive_metrics(frame)
        self.assertEqual(derived.hallucination_proxy[:4].tolist(), [0, 1, 1, 0])
        self.assertEqual(derived.retrieval_failure_proxy[:4].tolist(), [0, 1, 0, 0])
        self.assertEqual(derived.confident_hallucination_proxy[:4].tolist(), [0, 1, 0, 0])
        self.assertTrue(np.isnan(derived.hallucination_proxy.iloc[4]))
        self.assertEqual(derived.output_throughput.iloc[0], 5)
        self.assertTrue(np.isnan(derived.output_throughput.iloc[1]))
        self.assertEqual(derived.answer_relevant.iloc[0], 0)  # raw disagreement preserved
        for col, value in dict(configuration='test', family='test', precision='BF16',
                               question_id='1', prompt_token_count=20, path_length=2,
                               failed=False, relevancy_score=.7, context_relevancy_score=.8).items():
            derived[col] = value
        derived['scored'] = derived.correct.notna()
        row = summarize(derived).iloc[0]
        for col in ['failure_given_context_miss', 'success_given_context', 'context_miss_share_among_errors']:
            self.assertEqual(row[col], .5)
        self.assertEqual(row.context_sensitivity, 0)
        self.assertEqual(row.scored, 4)
        self.assertEqual(row.judge_disagreement_count, 1)
        only_hits = summarize(derived[derived.context_relevant.eq(1)]).iloc[0]
        self.assertTrue(np.isnan(only_hits.failure_given_context_miss))
        self.assertTrue(np.isnan(only_hits.context_sensitivity))
        normalized = radar_values(pd.DataFrame({'output_throughput': [2., 4.],
                                               'mean_path_length': [0., 0.], 'mean_latency': [5., 10.]}))
        self.assertEqual(normalized.output_throughput_normalized.tolist(), [.5, 1])
        self.assertEqual(normalized.mean_path_length_normalized.tolist(), [0, 0])
        self.assertEqual(PERFORMANCE_RADAR[4][0], 'output_throughput_normalized')
        self.assertEqual(PERFORMANCE_RADAR[5][0], 'mean_path_length_normalized')

    def test_matrix(self):
        runs = manifest('/tmp', 'test')
        self.assertEqual(len(runs), 10)
        self.assertEqual(len({r['results'] for r in runs}), 10)

    def test_missing_duplicate_invalid_and_paired(self):
        with tempfile.TemporaryDirectory() as folder:
            runs = manifest(folder, 'test')
            data, audit = load_runs(runs)
            self.assertTrue(data.empty)
            self.assertEqual(audit.simulation_rows.sum(), 0)
            for index in [0, 9]:
                path = runs[index]['results']
                path.mkdir(parents=True)
                sim = [dict(question_id=i, model_response='answer' if i != 2 else '',
                            processing_time=10 + index, processing_path=['start']) for i in [1, 2, 3]]
                (path / 'TC_OUTPUT_TC1.json').write_text(json.dumps(sim))
                ev = [dict(id=1, correct=index == 9, answer_relevant=True, context_relevant=False,
                           relevancy_score=.7, context_relevancy_score=.2),
                      dict(id=2, correct=False, answer_relevant=False, context_relevant=False,
                           correctness_reason='Judge JSON could not be parsed.')]
                (path / 'evaluation').mkdir()
                (path / 'evaluation/TC_EVAL_TC1.json').write_text(json.dumps(ev))
            data, audit = load_runs(runs)
            self.assertEqual(len(data), 6)
            self.assertEqual(data.scored.sum(), 2)
            self.assertEqual(data.invalid_judge.sum(), 2)
            self.assertEqual(data.failed.sum(), 2)
            self.assertEqual(data.relevancy_score.dropna().tolist(), [.7, .7])
            self.assertEqual(summarize(data).accuracy.tolist(), [0, 1])
            diff = paired(data)
            quality = diff[diff.metric.eq('correct')].iloc[0]
            self.assertEqual(quality.n, 1)
            self.assertEqual(quality.difference, -1)
            target = runs[0]['results'] / 'TC_OUTPUT_TC1.json'
            target.write_text(json.dumps([dict(question_id=1), dict(question_id=1)]))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_runs(runs)


if __name__ == '__main__':
    unittest.main()
