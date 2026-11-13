import contextlib
import io
import unittest
from unittest.mock import patch

from bot import answer


class AnswerTests(unittest.TestCase):
    def test_exact_names_rank_above_generic_text(self):
        docs = [{"id": "a", "text": "Makerspace A laser cutter PVC teflon"},
                {"id": "b", "text": "Engineering innovation workshop general news"}]
        self.assertEqual(answer.lexical_rank("laser cutter banned PVC", docs)[0]["id"], "a")

    def test_decompose_preserves_original_and_both_sides(self):
        q = "Compare Makerspace A and Makerspace B"
        self.assertEqual(answer.decompose(q), [q, "Compare Makerspace A", "Makerspace B"])
        self.assertEqual(answer.decompose("What are PVC and teflon?"), ["What are PVC and teflon?"])

    def test_answer_grounded_and_single_line(self):
        chunks = [{"text": "FDM means fused deposition modelling", "metadata": {"url": "source"}}]
        with patch.object(answer, "retrieve", return_value=chunks), patch.object(answer, "chat", return_value="Fused\ndeposition modelling") as chat:
            self.assertEqual(answer.rag_answer("What is FDM?"), "Fused deposition modelling")
            self.assertIn("source", chat.call_args.args[0][1]["content"])
            self.assertIn("FDM means", chat.call_args.args[0][1]["content"])

    def test_batch_preserves_successes_and_order(self):
        with patch.object(answer, "rag_answer", side_effect=["first", RuntimeError("offline"), "third"]), contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(answer.rag_answer_batch(["a", "b", "c"]), ["first", "", "third"])
            self.assertIn("offline", errors.getvalue())

    def test_count_parser_does_not_take_first_of_multiple_numbers(self):
        self.assertEqual(answer._format_answer("How many printers?", "There are 7 printers."), "7")
        self.assertEqual(answer._format_answer("How many printers?", "4 plus 3 is 7"), "4 plus 3 is 7")
        self.assertEqual(answer._format_answer("How many more posters?", "-8"), "-8")

    def test_empty_index_never_calls_embedding(self):
        with patch.object(answer, "get_store") as store, patch.object(answer, "_documents", return_value=()), patch.object(answer, "embed") as embed:
            store.return_value.count.return_value = 0
            self.assertEqual(answer.retrieve("hello"), [])
            embed.assert_not_called()

    def test_where_filter(self):
        self.assertTrue(answer._matches({"kind": "image", "year": "2025"},
                        {"$and": [{"kind": "image"}, {"year": "2025"}]}))
        self.assertFalse(answer._matches({"year": "2024"}, {"year": "2025"}))

    def test_inventory_reads_all_matches_and_deduplicates_images(self):
        from unittest.mock import MagicMock
        store = MagicMock()
        # Deliberately larger than k=5; count must not use nearest-neighbour search.
        metas = [{"image": f"poster-{i}", "url": "https://innoacademy.engg.hku.hk/workshop/"}
                 for i in range(31)]
        store.get.return_value = {"metadatas": metas + [metas[0]]}
        with patch.object(answer, "get_store", return_value=store), patch.object(answer, "_local_image_records", return_value=[]):
            evidence = answer.inventory_counts("How many workshop posters in Innovation Academy in 2025?")
        self.assertIn("31 unique", evidence)
        self.assertIn("coverage", evidence)
        store.query.assert_not_called()
        self.assertIn({"topic": "workshop"}, store.get.call_args.kwargs["where"]["$and"])

    def test_full_local_records_work_with_empty_dense_store(self):
        record = {"id": "local", "text": "Laser safety prohibits toxic fumes", "metadata": {"url": "manual"}}
        with patch.object(answer, "get_store") as store, patch.object(answer, "_documents", return_value=(record,)), patch.object(answer, "embed") as embed:
            store.return_value.count.return_value = 0
            self.assertEqual(answer.retrieve("laser safety")[0]["id"], "local")
            embed.assert_not_called()

    def test_semantic_failure_preserves_keyword_evidence(self):
        record = {"id": "local", "text": "Laser safety prohibits toxic fumes", "metadata": {"url": "manual"}}
        with patch.object(answer, "get_store") as store, patch.object(answer, "_documents", return_value=(record,)), patch.object(answer, "embed", side_effect=RuntimeError("quota max=100, actual=rejected; wait")), contextlib.redirect_stderr(io.StringIO()) as errors:
            store.return_value.count.return_value = 1
            self.assertEqual(answer.retrieve("laser safety")[0]["id"], "local")
            self.assertIn("max=100", errors.getvalue())

    def test_terminology_uses_general_knowledge_without_irrelevant_context(self):
        with patch.object(answer, "retrieve") as retrieve, patch.object(answer, "chat", return_value="definition"):
            self.assertEqual(answer.rag_answer("What is additive manufacturing also known as?"), "definition")
            retrieve.assert_not_called()

    def test_retrieved_page_expands_to_its_processed_image(self):
        text = {"id": "page", "text": "Laser cutter materials safety", "metadata": {"url": "manual"}}
        image = {"id": "image", "text": "VISIBLE TEXT: Toxic fumes; PVC not permitted", "metadata": {"url": "manual", "has_visual_description": True}}
        with patch.object(answer, "get_store") as store, patch.object(answer, "_documents", return_value=(text, image)):
            store.return_value.count.return_value = 0
            results = answer.retrieve("laser cutter safety", k=1)
        self.assertEqual([record["id"] for record in results], ["image", "page"])

    def test_invalid_k_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "min=1, asked for 0"):
            answer.retrieve("hello", k=0)


if __name__ == "__main__":
    unittest.main()
