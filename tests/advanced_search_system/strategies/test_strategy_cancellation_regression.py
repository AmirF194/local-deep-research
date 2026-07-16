"""Behavioural regression tests for strategy cancellation checks (#4871, PRs #4872/#5092).

Each test pins ONE specific ``check_termination()`` call site: the test
fails if that call site alone is deleted. The callback helper counts only
``termination_check``-phase invocations because in production every
progress emit is also a cancellation point (research_service's callback
raises on any invocation once Stop is requested) — a test that cancels on
an arbitrary emit keeps passing even with the strategy-level checks
removed, which is exactly the false confidence these tests must not give.
"""

from unittest.mock import MagicMock, patch
import pytest


def _cancel_on_nth_termination_check(n, arm_on_phase=None):
    """Build a progress callback that raises on the n-th termination check.

    Only callbacks with metadata phase ``termination_check`` count toward
    ``n``; every phase seen is recorded in ``state["phases"]`` so tests can
    assert which progress emits did (not) happen before the cancellation.

    arm_on_phase: when given, termination checks are ignored until a
    callback with that phase has been observed — used to target a check
    that sits after a known progress emit.
    """
    from local_deep_research.exceptions import ResearchTerminatedException

    state = {"checks": 0, "phases": [], "armed": arm_on_phase is None}

    def callback(message, progress_percent, metadata):
        phase = (metadata or {}).get("phase")
        state["phases"].append(phase)
        if arm_on_phase is not None and phase == arm_on_phase:
            state["armed"] = True
        if phase == "termination_check" and state["armed"]:
            state["checks"] += 1
            if state["checks"] == n:
                raise ResearchTerminatedException("cancelled")

    return callback, state


def _make_source_based_strategy(settings_snapshot=None):
    from local_deep_research.advanced_search_system.strategies.source_based_strategy import (
        SourceBasedSearchStrategy,
    )

    return SourceBasedSearchStrategy(
        model=MagicMock(),
        search=MagicMock(),
        all_links_of_system=[],
        settings_snapshot=settings_snapshot,
    )


class TestSourceBasedCancellation:
    def test_cancel_at_loop_entry_runs_no_llm(self):
        """Pins the loop-entry check (PR #4872): cancel before iteration 1
        generates questions."""
        from local_deep_research.exceptions import ResearchTerminatedException

        strategy = _make_source_based_strategy()
        callback, state = _cancel_on_nth_termination_check(n=1)
        strategy.set_progress_callback(callback)
        gen_mock = MagicMock(return_value=["Q1"])
        with (
            patch.object(
                strategy.question_generator, "generate_questions", gen_mock
            ),
            patch(
                "local_deep_research.advanced_search_system.strategies.source_based_strategy.run_parallel_searches",
                return_value=[],
            ),
            patch.object(
                strategy.citation_handler,
                "analyze_followup",
                return_value={"content": "Done", "documents": []},
            ),
        ):
            with pytest.raises(ResearchTerminatedException):
                strategy.analyze_topic("test query")
        assert state["checks"] == 1
        assert gen_mock.call_count == 0

    def test_cancel_after_parallel_search_fires_before_result_processing(
        self,
    ):
        """Pins the post-search check (PR #5092): with a single iteration,
        termination check #2 is the one right after run_parallel_searches —
        if it were removed, the next check comes only after the cross-engine
        filter has already run and emitted its progress."""
        from local_deep_research.exceptions import ResearchTerminatedException

        strategy = _make_source_based_strategy(
            settings_snapshot={"search.iterations": 1}
        )
        callback, state = _cancel_on_nth_termination_check(n=2)
        strategy.set_progress_callback(callback)
        gen_mock = MagicMock(return_value=["Q1"])
        filter_mock = MagicMock(return_value=[])
        synth_mock = MagicMock(
            return_value={"content": "Done", "documents": []}
        )
        with (
            patch.object(
                strategy.question_generator, "generate_questions", gen_mock
            ),
            patch(
                "local_deep_research.advanced_search_system.strategies.source_based_strategy.run_parallel_searches",
                return_value=[],
            ),
            patch.object(
                strategy.cross_engine_filter, "filter_results", filter_mock
            ),
            patch.object(
                strategy.citation_handler, "analyze_followup", synth_mock
            ),
        ):
            with pytest.raises(ResearchTerminatedException):
                strategy.analyze_topic("test query")
        assert state["checks"] == 2
        # Iteration 1 ran (questions were generated) …
        assert gen_mock.call_count == 1
        # … but the cancel fired before any post-search work: the
        # cross-engine filter never ran and never emitted progress.
        assert filter_mock.call_count == 0
        assert "final_filtering" not in state["phases"]
        assert synth_mock.call_count == 0

    def test_cancel_before_synthesis_aborts_llm_call(self):
        """Pins the pre-synthesis check (PR #5092): the first termination
        check after the filtering-complete emit must fire before the
        synthesis emit / citation-handler LLM call."""
        from local_deep_research.exceptions import ResearchTerminatedException

        strategy = _make_source_based_strategy(
            settings_snapshot={"search.iterations": 1}
        )
        callback, state = _cancel_on_nth_termination_check(
            n=1, arm_on_phase="filtering_complete"
        )
        strategy.set_progress_callback(callback)
        filter_mock = MagicMock(return_value=[])
        synth_mock = MagicMock(
            return_value={"content": "Done", "documents": []}
        )
        with (
            patch.object(
                strategy.question_generator,
                "generate_questions",
                return_value=["Q1"],
            ),
            patch(
                "local_deep_research.advanced_search_system.strategies.source_based_strategy.run_parallel_searches",
                return_value=[],
            ),
            patch.object(
                strategy.cross_engine_filter, "filter_results", filter_mock
            ),
            patch.object(
                strategy.citation_handler, "analyze_followup", synth_mock
            ),
        ):
            with pytest.raises(ResearchTerminatedException):
                strategy.analyze_topic("test query")
        assert state["checks"] == 1
        # We really got past the filtering phase …
        assert filter_mock.call_count == 1
        # … and cancelled before synthesis started.
        assert synth_mock.call_count == 0
        assert "synthesis" not in state["phases"]


# NewsAggregationStrategy
def _make_news_strategy():
    from local_deep_research.advanced_search_system.strategies.news_strategy import (
        NewsAggregationStrategy,
    )

    return NewsAggregationStrategy(model=MagicMock(), search=MagicMock())


class TestNewsStrategyCancellation:
    def test_cancel_before_work_runs_no_llm_or_search(self):
        """Pins the top-of-analyze_topic check: cancel before question
        generation (the first LLM call) and before any search."""
        from local_deep_research.exceptions import ResearchTerminatedException

        strategy = _make_news_strategy()
        callback, state = _cancel_on_nth_termination_check(n=1)
        strategy.set_progress_callback(callback)
        gen_mock = MagicMock(return_value=["Q1"])
        search_mock = MagicMock(return_value=[])
        strategy.search.run = search_mock
        strategy.model.invoke = MagicMock(return_value=MagicMock(content="{}"))
        with patch.object(
            strategy.question_generator, "generate_questions", gen_mock
        ):
            with pytest.raises(ResearchTerminatedException):
                strategy.analyze_topic("news query")
        assert state["checks"] == 1
        assert gen_mock.call_count == 0
        assert search_mock.call_count == 0

    def test_cancel_before_analysis_llm_stops_synthesis(self):
        """Pins the pre-analysis check: termination check #2 sits between
        the search loop and the analysis LLM call."""
        from local_deep_research.exceptions import ResearchTerminatedException

        strategy = _make_news_strategy()
        callback, state = _cancel_on_nth_termination_check(n=2)
        strategy.set_progress_callback(callback)
        search_mock = MagicMock(
            return_value=[{"title": "T", "snippet": "S", "link": "http://x"}]
        )
        strategy.search.run = search_mock
        model_invoke = MagicMock(return_value=MagicMock(content="{}"))
        strategy.model.invoke = model_invoke
        with patch.object(
            strategy.question_generator,
            "generate_questions",
            return_value=["Q1"],
        ):
            with pytest.raises(ResearchTerminatedException):
                strategy.analyze_topic("news query")
        assert state["checks"] == 2
        # The searches ran …
        assert search_mock.call_count == 1
        # … but the analysis LLM call was never made.
        assert model_invoke.call_count == 0


# TopicOrganizationStrategy
def _make_topic_org_strategy():
    from local_deep_research.advanced_search_system.strategies.topic_organization_strategy import (
        TopicOrganizationStrategy,
    )

    return TopicOrganizationStrategy(
        search=MagicMock(), model=MagicMock(), all_links_of_system=[]
    )


class TestTopicOrganizationCancellation:
    def test_progress_callback_propagates_to_delegate(self):
        """Pins the set_progress_callback override: without propagation the
        delegate's check_termination() calls are silent no-ops and a Stop
        during source gathering is not detected until the delegate ends."""
        strategy = _make_topic_org_strategy()

        def callback(message, progress_percent, metadata):
            pass

        strategy.set_progress_callback(callback)
        assert strategy.source_strategy.progress_callback is callback

    def test_cancel_during_delegate_gathering_stops_delegate(self):
        """End-to-end: with the callback propagated, termination check #2
        is the delegate's own loop-entry check — the delegate stops before
        generating any questions."""
        from local_deep_research.exceptions import ResearchTerminatedException

        strategy = _make_topic_org_strategy()
        callback, state = _cancel_on_nth_termination_check(n=2)
        strategy.set_progress_callback(callback)
        delegate_gen_mock = MagicMock(return_value=["Q1"])
        with (
            patch.object(
                strategy.source_strategy.question_generator,
                "generate_questions",
                delegate_gen_mock,
            ),
            patch(
                "local_deep_research.advanced_search_system.strategies.source_based_strategy.run_parallel_searches",
                return_value=[],
            ),
        ):
            with pytest.raises(ResearchTerminatedException):
                strategy.analyze_topic("topic query")
        assert state["checks"] == 2
        assert delegate_gen_mock.call_count == 0

    def test_cancel_before_delegation_runs_no_source_strategy(self):
        """Pins the top-of-analyze_topic check: cancel before the delegate
        is invoked at all."""
        from local_deep_research.exceptions import ResearchTerminatedException

        strategy = _make_topic_org_strategy()
        callback, state = _cancel_on_nth_termination_check(n=1)
        strategy.set_progress_callback(callback)
        delegate_mock = MagicMock(
            return_value={
                "all_links_of_system": [],
                "iterations": 0,
                "questions_by_iteration": {},
            }
        )
        strategy.source_strategy = MagicMock()
        strategy.source_strategy.analyze_topic = delegate_mock
        with pytest.raises(ResearchTerminatedException):
            strategy.analyze_topic("topic query")
        assert state["checks"] == 1
        assert delegate_mock.call_count == 0

    def test_cancel_in_refinement_loop_stops_before_question_generation(
        self,
    ):
        """Pins the refinement-loop-top check: the flow genuinely reaches
        the refinement loop (delegate returns sources, topics extracted),
        and termination check #2 fires before the refinement-question LLM
        call."""
        from local_deep_research.exceptions import ResearchTerminatedException

        strategy = _make_topic_org_strategy()
        strategy.enable_refinement = True
        strategy.max_refinement_iterations = 2
        strategy.refinement_questions = []
        strategy.topic_graph = MagicMock()
        # Replace the delegate BEFORE wiring the callback so the mock
        # absorbs the propagation; the delegate returns one source so the
        # no-sources early return is not taken.
        delegate_mock = MagicMock(
            return_value={
                "all_links_of_system": [
                    {"title": "T", "snippet": "S", "link": "http://x"}
                ],
                "iterations": 1,
                "questions_by_iteration": {},
            }
        )
        strategy.source_strategy = MagicMock()
        strategy.source_strategy.analyze_topic = delegate_mock
        callback, state = _cancel_on_nth_termination_check(n=2)
        strategy.set_progress_callback(callback)
        topic_mock = MagicMock()
        topic_mock.title = "Stub topic"
        topic_mock.get_all_sources = MagicMock(
            return_value=[{"link": "http://x"}]
        )
        # Finite side_effect ending in None so the loop can never spin
        # forever, whatever happens to the checks under test.
        refinement_gen_mock = MagicMock(side_effect=["another question", None])
        with (
            patch.object(
                strategy,
                "_extract_topics_from_sources",
                return_value=[topic_mock],
            ),
            patch.object(
                strategy, "_generate_refinement_question", refinement_gen_mock
            ),
            patch.object(strategy, "_find_topic_relationships"),
        ):
            with pytest.raises(ResearchTerminatedException):
                strategy.analyze_topic("topic query")
        assert state["checks"] == 2
        # We got as far as the delegate …
        assert delegate_mock.call_count == 1
        # … and the cancel fired at the top of the refinement loop, before
        # the refinement-question LLM call.
        assert refinement_gen_mock.call_count == 0

    def test_cancel_during_refinement_gathering_stops_refinement_delegate(
        self,
    ):
        """Pins the callback propagation to the per-iteration refinement
        strategy: termination check #3 is the refinement delegate's own
        loop-entry check, so it stops before generating any questions."""
        from local_deep_research.exceptions import ResearchTerminatedException

        strategy = _make_topic_org_strategy()
        strategy.enable_refinement = True
        strategy.max_refinement_iterations = 2
        strategy.refinement_questions = []
        strategy.topic_graph = MagicMock()
        delegate_mock = MagicMock(
            return_value={
                "all_links_of_system": [
                    {"title": "T", "snippet": "S", "link": "http://x"}
                ],
                "iterations": 1,
                "questions_by_iteration": {},
            }
        )
        strategy.source_strategy = MagicMock()
        strategy.source_strategy.analyze_topic = delegate_mock
        callback, state = _cancel_on_nth_termination_check(n=3)
        strategy.set_progress_callback(callback)
        topic_mock = MagicMock()
        topic_mock.title = "Stub topic"
        topic_mock.get_all_sources = MagicMock(
            return_value=[{"link": "http://x"}]
        )
        refinement_gen_mock = MagicMock(side_effect=["another question", None])
        # The refinement strategy is constructed inside the loop, so its
        # question generator can only be intercepted at the class level.
        refinement_question_gen = MagicMock(return_value=["Q1"])
        with (
            patch.object(
                strategy,
                "_extract_topics_from_sources",
                return_value=[topic_mock],
            ),
            patch.object(
                strategy, "_generate_refinement_question", refinement_gen_mock
            ),
            patch.object(strategy, "_find_topic_relationships"),
            patch(
                "local_deep_research.advanced_search_system.questions.standard_question.StandardQuestionGenerator.generate_questions",
                refinement_question_gen,
            ),
            patch(
                "local_deep_research.advanced_search_system.strategies.source_based_strategy.run_parallel_searches",
                return_value=[],
            ),
        ):
            with pytest.raises(ResearchTerminatedException):
                strategy.analyze_topic("topic query")
        assert state["checks"] == 3
        # A refinement question was generated (the loop genuinely started) …
        assert refinement_gen_mock.call_count == 1
        # … and the refinement delegate stopped at its own loop-entry
        # check, before its question-generation LLM call.
        assert refinement_question_gen.call_count == 0


# EnhancedContextualFollowupStrategy
def _make_followup_strategy():
    from local_deep_research.advanced_search_system.strategies.followup.enhanced_contextual_followup import (
        EnhancedContextualFollowUpStrategy,
    )

    delegate = MagicMock()
    delegate.analyze_topic = MagicMock(
        return_value={
            "findings": [],
            "iterations": 0,
            "questions_by_iteration": {},
            "formatted_findings": "",
            "current_knowledge": "",
            "all_links_of_system": [],
            "error": None,
        }
    )
    return EnhancedContextualFollowUpStrategy(
        model=MagicMock(),
        search=MagicMock(),
        delegate_strategy=delegate,
        all_links_of_system=[],
    )


class TestEnhancedContextualFollowupCancellation:
    def test_cancel_before_contextualized_query_runs_no_llm(self):
        """Pins the top-of-analyze_topic check: cancel before the
        contextualized-query LLM call and the delegate research."""
        from local_deep_research.exceptions import ResearchTerminatedException

        strategy = _make_followup_strategy()
        callback, state = _cancel_on_nth_termination_check(n=1)
        strategy.set_progress_callback(callback)
        contextualized_query_mock = MagicMock(return_value="rewritten query")
        strategy.question_generator.generate_contextualized_query = (
            contextualized_query_mock
        )
        with pytest.raises(ResearchTerminatedException):
            strategy.analyze_topic("follow-up query")
        assert state["checks"] == 1
        assert contextualized_query_mock.call_count == 0
        assert strategy.delegate_strategy.analyze_topic.call_count == 0


# LangGraphAgentStrategy fallback synthesis
def _make_langgraph_strategy():
    from local_deep_research.advanced_search_system.strategies.langgraph_agent_strategy import (
        LangGraphAgentStrategy,
    )

    return LangGraphAgentStrategy(
        model=MagicMock(), search=MagicMock(), all_links_of_system=[]
    )


class TestLangGraphFallbackSynthesisCancellation:
    def test_cancel_before_fallback_synthesis_aborts_llm_call(self):
        """Pins the check at the top of _synthesize_from_collector: cancel
        before the fallback synthesis LLM call."""
        from local_deep_research.exceptions import ResearchTerminatedException

        strategy = _make_langgraph_strategy()
        # collector.add_results (its public API) populates the internal
        # _results list that the read-only .results property exposes.
        strategy.collector.add_results(
            [{"title": "T", "snippet": "S", "link": "http://x"}],
            engine_name="web",
        )
        callback, state = _cancel_on_nth_termination_check(n=1)
        strategy.set_progress_callback(callback)
        model_invoke = MagicMock(
            return_value=MagicMock(content="synthesized answer")
        )
        strategy.model.invoke = model_invoke
        with pytest.raises(ResearchTerminatedException):
            strategy._synthesize_from_collector("test query")
        assert state["checks"] == 1
        assert model_invoke.call_count == 0
