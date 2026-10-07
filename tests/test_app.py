import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

from streamlit.testing.v1 import AppTest
from tests.ui.fixtures import execution_result


APP = str(Path(__file__).resolve().parents[1] / "app.py")


class AppTests(unittest.TestCase):
    def setUp(self):
        self.service = Mock()
        self.service.count.return_value = 2
        self.service.ingest_file.return_value = 1
        default = patch("rag.context.get_default_rag_service", return_value=self.service)
        default.start()
        self.addCleanup(default.stop)

    def test_controls_toggle_canonical_research_and_result_rendering(self):
        result = execution_result()
        result.final_answer = "Answer [Document: notes.md]"
        with patch("agents.run_research_detailed", return_value=result) as research:
            app = AppTest.from_file(APP, default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("file_uploader")), 1)
            self.assertTrue(app.checkbox(key="use_rag").value)
            self.assertTrue(app.toggle(key="use_cache").value)
            self.assertEqual(app.selectbox(key="model_route").value, "Auto")
            self.assertEqual(app.selectbox(key="model_route").options, ["Auto", "Fast", "Quality"])
            self.assertIn("Stored chunks: 2", [c.value for c in app.caption])
            self.service.ingest_file.assert_not_called()
            app.checkbox(key="use_rag").uncheck().run()
            app.toggle(key="use_cache").set_value(False).run()
            app.text_area(key="research_query").set_value("codename")
            next(b for b in app.button if b.label == "Research").click().run()
            self.assertFalse(app.exception)
            research.assert_called_once_with("codename", use_rag=False, use_cache=False, model_route="auto",
                                             progress_callback=ANY)
            self.assertTrue(callable(research.call_args.kwargs['progress_callback']))
            self.assertIn("Answer [Document: notes.md]", [m.value for m in app.markdown])
            self.assertIs(app.session_state.research_execution, result)

    def test_manual_routes_reach_canonical_backend_without_automatic_research(self):
        for label in ("Fast", "Quality"):
            with self.subTest(route=label), patch("agents.run_research_detailed", return_value=execution_result()) as research:
                app = AppTest.from_file(APP, default_timeout=30).run()
                app.selectbox(key="model_route").select(label).run()
                self.assertFalse(app.exception)
                research.assert_not_called()
                app.text_area(key="research_query").set_value("What is MCP?")
                next(b for b in app.button if b.label == "Research").click().run()
                self.assertFalse(app.exception)
                research.assert_called_once_with("What is MCP?", use_rag=True, use_cache=True,
                                                 model_route=label.lower(), progress_callback=ANY)

    def test_explicit_ingestion_and_cleanup_on_success_and_failure(self):
        upload = SimpleNamespace(name="notes.md", getvalue=lambda: b"protocol notes")
        for fail in (False, True):
            with self.subTest(fail=fail), patch("streamlit.file_uploader", return_value=[upload]):
                paths = []

                def ingest(path, *, source):
                    paths.append(path)
                    self.assertEqual(source, "notes.md")
                    self.assertEqual(path.read_bytes(), b"protocol notes")
                    if fail:
                        raise RuntimeError("embedding offline")
                    return 1

                self.service.ingest_file.reset_mock()
                self.service.ingest_file.side_effect = ingest
                app = AppTest.from_file(APP, default_timeout=30).run()
                self.service.ingest_file.assert_not_called()
                app.button(key="add_documents").click().run()
                self.assertFalse(app.exception)
                self.service.ingest_file.assert_called_once()
                self.assertFalse(paths[0].exists())
                self.assertTrue(app.error if fail else app.success)
                app.run()
                self.service.ingest_file.assert_called_once()

    def test_unavailable_corpus_preserves_research_ui(self):
        with patch("rag.context.get_default_rag_service", side_effect=RuntimeError("store offline")):
            app = AppTest.from_file(APP, default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.warning)
        self.assertEqual(len(app.text_area), 1)

    def test_error_and_exception_do_not_render_stale_answer(self):
        for error in (None, RuntimeError('backend offline')):
            with self.subTest(error=error), patch('agents.run_research_detailed',
                    return_value=execution_result('error'), side_effect=error):
                app = AppTest.from_file(APP, default_timeout=30).run()
                app.session_state.research_result = 'stale answer'
                app.text_area(key='research_query').set_value('MCP')
                next(b for b in app.button if b.label == 'Research').click().run()
                self.assertFalse(app.exception)
                self.assertTrue(app.error)
                self.assertIsNone(app.session_state.research_result)

    def test_result_survives_rerun_navigation_refresh_and_control_changes(self):
        result = execution_result('complex')
        with patch('agents.run_research_detailed', return_value=result) as research:
            app = AppTest.from_file(APP, default_timeout=30).run()
            self.assertEqual([tab.label for tab in app.tabs], ['Research', 'Analytics'])
            app.text_area(key='research_query').set_value('compare protocols')
            next(b for b in app.button if b.label == 'Research').click().run()
            app.run()
            app.session_state.console_view = 'Analytics'
            app.run()
            app.button(key='refresh_analytics').click().run()
            app.session_state.console_view = 'Research'
            app.selectbox(key='model_route').select('Quality').run()
            self.assertFalse(app.exception)
            research.assert_called_once()
            self.assertIs(app.session_state.research_execution, result)
            self.assertIn(result.final_answer, [m.value for m in app.markdown])
            self.assertEqual(app.text_area(key='research_query').value, 'compare protocols')
            self.service.ingest_file.assert_not_called()

    def test_analytics_failure_does_not_prevent_explicit_research(self):
        with patch('observability.store.get_default_metrics_store', side_effect=OSError('SECRET')), \
             patch('agents.run_research_detailed', return_value=execution_result()) as research:
            app = AppTest.from_file(APP, default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)
            app.text_area(key='research_query').set_value('MCP')
            next(b for b in app.button if b.label == 'Research').click().run()
            self.assertFalse(app.exception)
            research.assert_called_once()
            self.assertIn('Postprocessed final research answer', '\n'.join(m.value for m in app.markdown))

    def test_empty_query_does_not_execute_research(self):
        with patch('agents.run_research_detailed') as research:
            app = AppTest.from_file(APP, default_timeout=30).run()
            app.text_area(key='research_query').set_value('   ')
            next(b for b in app.button if b.label == 'Research').click().run()
            self.assertTrue(app.warning)
            research.assert_not_called()


if __name__ == "__main__":
    unittest.main()
