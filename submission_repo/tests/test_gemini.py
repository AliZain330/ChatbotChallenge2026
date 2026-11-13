"""REST-contract tests; no live or paid API calls."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from bot import gemini


def response(payload, status=200):
    result = MagicMock()
    result.status_code = status
    result.json.return_value = payload
    return result


def completion(text='answer', reason='STOP'):
    return {'candidates': [{'finishReason': reason, 'content': {'parts': [{'text': text}]}}]}


class GeminiTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'GEMINI_API_KEY': 'test-secret-never-log'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_chat_translation_and_header_key(self):
        with patch.object(gemini.requests, 'request', return_value=response(completion())) as request:
            result = gemini.chat([{'role': 'system', 'content': 'grounded'},
                                  {'role': 'user', 'content': 'first'},
                                  {'role': 'assistant', 'content': 'reply'},
                                  {'role': 'user', 'content': 'next'}], 'model-name', 0.0, 512, 30)
        self.assertEqual(result, 'answer')
        args, kwargs = request.call_args
        self.assertEqual(args[1], gemini.BASE_URL + '/models/model-name:generateContent')
        self.assertNotIn('test-secret', args[1])
        self.assertEqual(kwargs['headers']['x-goog-api-key'], 'test-secret-never-log')
        self.assertFalse(kwargs['allow_redirects'])
        self.assertEqual(kwargs['json']['contents'][1]['role'], 'model')
        self.assertEqual(kwargs['json']['systemInstruction']['parts'], [{'text': 'grounded'}])
        self.assertNotIn('tools', kwargs['json'])

    def test_thought_parts_excluded(self):
        result = {'candidates': [{'content': {'parts': [{'thought': True, 'text': 'private'}, {'text': 'public'}]}, 'finishReason': 'STOP'}]}
        self.assertEqual(gemini._text(result), 'public')

    def test_empty_blocked_truncated_fail(self):
        for payload in [{}, {'promptFeedback': {'blockReason': 'SAFETY'}}, completion('', 'STOP'), completion('partial', 'MAX_TOKENS'), completion('text', 'SAFETY')]:
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                gemini._text(payload)

    def test_errors_do_not_echo_secret(self):
        with patch.object(gemini.requests, 'request', return_value=response({'error': {'message': 'test-secret-never-log'}}, 429)):
            with self.assertRaisesRegex(RuntimeError, 'actual=429') as caught:
                gemini._request('POST', 'models/example:generateContent', 30, {})
        self.assertNotIn('test-secret', str(caught.exception))
        with patch.object(gemini.requests, 'request', side_effect=gemini.requests.ConnectionError('test-secret-never-log')):
            with self.assertRaises(RuntimeError) as caught:
                gemini._request('POST', 'models/example:generateContent', 30, {})
        self.assertNotIn('test-secret', str(caught.exception))

    def test_missing_key_timeout_no_network(self):
        with patch.object(gemini.requests, 'request') as request:
            with patch.dict(os.environ, {'GEMINI_API_KEY': ''}):
                with self.assertRaisesRegex(RuntimeError, 'actual=missing'):
                    gemini._request('GET', 'models', 30)
            with self.assertRaises(TimeoutError):
                gemini._request('GET', 'models', 0)
            request.assert_not_called()

    def test_embedding_batches_task_order(self):
        batches = []
        def fake(method, resource, timeout, payload):
            batches.append(payload)
            return {'embeddings': [{'values': [float(item['content']['parts'][0]['text']), 0.5]} for item in payload['requests']]}
        texts = [str(i) for i in range(gemini.MAX_EMBED_BATCH + 1)]
        with patch.object(gemini, '_request', side_effect=fake):
            vectors = gemini.embed(texts, 'models/embedding-model', 256, 30, 'RETRIEVAL_QUERY')
        self.assertEqual([len(b['requests']) for b in batches], [100, 1])
        self.assertEqual([v[0] for v in vectors], [float(i) for i in range(len(texts))])
        self.assertTrue(all(r['taskType'] == 'RETRIEVAL_QUERY' for b in batches for r in b['requests']))
        self.assertTrue(all(r['model'] == 'models/embedding-model' for b in batches for r in b['requests']))

    def test_bad_vector_count_dimensions_nonfinite(self):
        for output in [[], [{'values': [1.0]}], [{'values': [1.0]}, {'values': [1.0, 2.0]}], [{'values': [float('nan')]}, {'values': [0.0]}], [{'values': [True]}, {'values': [0.0]}]]:
            with self.subTest(output=output), patch.object(gemini, '_request', return_value={'embeddings': output}):
                with self.assertRaises(RuntimeError):
                    gemini.embed(['a', 'b'], 'embedding-model', 100, 30)

    def test_invalid_model_no_endpoint_override(self):
        for model in ['https://evil.example', '../escape', 'models/a?key=secret', '']:
            with self.assertRaises(ValueError):
                gemini._model_name(model)

    def test_inline_vision_bytes_mime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            path.write_bytes(b'example')
            with patch.object(gemini, '_request', return_value=completion('visible sign')) as request:
                self.assertEqual(gemini.describe_image(str(path), 'transcribe', 'vision-model', 30), 'visible sign')
            body = request.call_args.args[3]
        image = body['contents'][0]['parts'][1]['inlineData']
        self.assertEqual(image['mimeType'], 'image/png')
        self.assertEqual(image['data'], 'ZXhhbXBsZQ==')
        self.assertNotIn('tools', body)

    def test_model_listing_pagination(self):
        pages = [{'models': [{'name': 'models/a', 'supportedGenerationMethods': ['generateContent'], 'description': 'unneeded'}], 'nextPageToken': 'next'}, {'models': [{'name': 'models/b', 'supportedGenerationMethods': ['embedContent']}]}]
        with patch.object(gemini, '_request', side_effect=pages) as request:
            models = gemini.list_models(30)
        self.assertEqual([m['name'] for m in models], ['models/a', 'models/b'])
        self.assertEqual(request.call_args.kwargs['params'], {'pageToken': 'next'})
        self.assertNotIn('description', models[0])

    def test_empty_embeddings_no_network(self):
        with patch.object(gemini, '_request') as request:
            self.assertEqual(gemini.embed([], 'embedding-model', 100, 30), [])
            request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
