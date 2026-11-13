"""Verify retained Azure routing and the shared grading time budget without API calls."""
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from bot import llm


class LLMTests(unittest.TestCase):
    def test_azure_chat_preserves_client_route(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="connected"))])
        with patch.object(llm, "PROVIDER", "azure"), patch.object(llm, "_request_client", return_value=client) as factory:
            self.assertEqual(llm.chat([{"role": "user", "content": "hello"}]), "connected")
        factory.assert_called_once_with(llm.CHAT_URL)
        self.assertIn("/chat/completions?api-version=", llm.CHAT_URL)

    def test_azure_embeddings_restore_input_order(self):
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(data=[
            SimpleNamespace(index=1, embedding=[0., 1.]), SimpleNamespace(index=0, embedding=[1., 0.])])
        with patch.object(llm, "PROVIDER", "azure"), patch.object(llm, "_request_client", return_value=client):
            self.assertEqual(llm.embed(["first", "second"]), [[1., 0.], [0., 1.]])
        self.assertIn("/openai/deployments/", llm.EMBED_URL)

    def test_question_budget_consumed_across_calls_and_reset(self):
        with patch.object(llm.time, "monotonic", side_effect=[100., 105., 131.]):
            with llm.question_budget():
                self.assertEqual(llm._timeout(), 25.)
                with self.assertRaisesRegex(TimeoutError, "max=30s"):
                    llm._timeout()
        self.assertEqual(llm._timeout(), 30.)

    def test_gemini_query_task_is_forwarded(self):
        with patch.object(llm, "PROVIDER", "gemini"), patch("bot.gemini.embed", return_value=[[1.]]) as embed:
            self.assertEqual(llm.embed(["query"], task_type="RETRIEVAL_QUERY"), [[1.]])
        self.assertEqual(embed.call_args.args[-1], "RETRIEVAL_QUERY")


if __name__ == "__main__":
    unittest.main()
