"""Keep successful fallback answers distinct from fully available retrieval."""
from contextlib import redirect_stderr, redirect_stdout
import io
import sys
import unittest
from unittest.mock import patch

from evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def test_fallback_warning_is_attached_to_the_correct_question(self):
        def answer(question):
            if question == 'fallback':
                print('[retrieval] semantic call failed: daily quota max=1000; wait for reset', file=sys.stderr)
            return 'correct'

        questions = [{'level': 2, 'question': q, 'answer': 'correct'} for q in ['normal', 'fallback']]
        with patch('evaluate.rag_answer', side_effect=answer), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
            result = evaluate(questions)
        self.assertEqual(result['completed'], 2)
        self.assertEqual(result['exact_matches'], 2)
        self.assertEqual(result['semantic_fallback_questions'], 1)
        self.assertFalse(result['results'][0]['semantic_fallback_observed'])
        self.assertTrue(result['results'][1]['semantic_fallback_observed'])
        self.assertIn('max=1000', result['results'][1]['diagnostics'][0])
        self.assertIn('max=1000', stderr.getvalue())

    def test_generation_failure_is_not_mislabeled_as_semantic_fallback(self):
        with patch('evaluate.rag_answer', side_effect=RuntimeError('generation unavailable')), redirect_stdout(io.StringIO()):
            result = evaluate([{'level': 1, 'question': 'question', 'answer': 'answer'}])
        self.assertEqual(result['completed'], 0)
        self.assertEqual(result['semantic_fallback_questions'], 0)
        self.assertIn('generation unavailable', result['results'][0]['error'])


if __name__ == '__main__':
    unittest.main()
