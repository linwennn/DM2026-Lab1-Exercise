import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
    # Instantiate single shared session state
    session = SessionState()

    # Load load_dataset_tool from premade_tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Load the fixture dataset into session
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build DTM using build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Record initial matrix and feature names to verify non-mutation
    orig_matrix = session.feature_matrix.copy()
    orig_feature_names = list(session.feature_names)

    # Initialize variance_filter_tool from tools_filtering
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(
        t for t in filtering_tools if t.name == "variance_filter_tool"
    )

    # Invoke variance_filter_tool with threshold=0.15 (Known Answer #5)
    result = variance_filter_tool.invoke({"threshold": 0.15})

    # Assert returned summary structure and threshold decisions
    assert result["threshold"] == 0.15
    assert result["total_terms"] == 5
    assert result["retained_terms"] == 4
    assert result["removed_terms"] == 1
    assert set(result["retained_list"]) == {"alpha", "beta", "delta", "gamma"}
    assert result["removed_list"] == ["always"]

    # Assert stored result and full_report
    stored = session.results_store[result["result_id"]]
    record = stored["record"]
    assert record.tool_name == "variance_filter_tool"

    full_report = stored["full_report"]
    assert full_report is not None
    assert list(full_report.columns) == ["term", "variance"]

    # Check that full_report is sorted descending by variance
    variances_in_report = full_report["variance"].tolist()
    assert variances_in_report == sorted(variances_in_report, reverse=True)

    # Check exact population variances per term (TEST_FIXTURE.md #5)
    expected_variances = {
        "alpha": 1.1875,
        "gamma": 1.109375,
        "delta": 0.25,
        "beta": 0.234375,
        "always": 0.109375,
    }
    report_map = dict(zip(full_report["term"], full_report["variance"]))
    for term, exp_var in expected_variances.items():
        assert report_map[term] == pytest.approx(exp_var, abs=1e-4)

    # Assert session feature_matrix and feature_names remain unchanged
    assert np.array_equal(session.feature_matrix.toarray(), orig_matrix.toarray())
    assert session.feature_names == orig_feature_names


def test_pearson_filter_tool_known_answer_6():
    # Instantiate single shared session state
    session = SessionState()

    # Load load_dataset_tool from premade_tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Load the fixture dataset into session
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build DTM using build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Record initial matrix and feature names to verify non-mutation
    orig_matrix = session.feature_matrix.copy()
    orig_feature_names = list(session.feature_names)

    # Initialize pearson_filter_tool from tools_filtering
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(
        t for t in filtering_tools if t.name == "pearson_filter_tool"
    )

    # Invoke pearson_filter_tool targeting catB with threshold=0.7
    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.7})

    # Expected correlations from TEST_FIXTURE.md #6 for target_class="catB"
    expected_r = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }

    # Assert returned summary structure and absolute threshold decisions (|r| > 0.7)
    assert result["target_class"] == "catB"
    assert result["threshold"] == 0.7
    assert result["total_terms"] == 5
    assert result["retained_terms"] == 3
    assert result["removed_terms"] == 2
    assert set(result["retained_list"]) == {"beta", "delta", "gamma"}
    assert set(result["removed_list"]) == {"alpha", "always"}

    # Assert stored result and full_report
    stored = session.results_store[result["result_id"]]
    record = stored["record"]
    assert record.tool_name == "pearson_filter_tool"

    full_report = stored["full_report"]
    assert full_report is not None
    assert list(full_report.columns) == ["term", "pearson_r"]

    # Check that full_report is sorted descending by pearson_r
    r_in_report = full_report["pearson_r"].tolist()
    assert r_in_report == sorted(r_in_report, reverse=True)

    # Check exact Pearson r values per term
    report_map = dict(zip(full_report["term"], full_report["pearson_r"]))
    for term, exp_val in expected_r.items():
        assert report_map[term] == pytest.approx(exp_val, abs=1e-3)

    # Assert session feature_matrix and feature_names remain unchanged
    assert np.array_equal(session.feature_matrix.toarray(), orig_matrix.toarray())
    assert session.feature_names == orig_feature_names


def test_spearman_filter_tool_known_answer_6():
    # Instantiate single shared session state
    session = SessionState()

    # Load load_dataset_tool from premade_tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Load the fixture dataset into session
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build DTM using build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Record initial matrix and feature names to verify non-mutation
    orig_matrix = session.feature_matrix.copy()
    orig_feature_names = list(session.feature_names)

    # Initialize spearman_filter_tool from tools_filtering
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(
        t for t in filtering_tools if t.name == "spearman_filter_tool"
    )

    # Invoke spearman_filter_tool targeting catB with threshold=0.7
    result = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.7})

    # Expected correlations from TEST_FIXTURE.md #6 for target_class="catB"
    expected_r = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }

    # Assert returned summary structure and absolute threshold decisions (|r| > 0.7)
    assert result["target_class"] == "catB"
    assert result["threshold"] == 0.7
    assert result["total_terms"] == 5
    assert result["retained_terms"] == 4
    assert result["removed_terms"] == 1
    assert set(result["retained_list"]) == {"alpha", "beta", "delta", "gamma"}
    assert set(result["removed_list"]) == {"always"}

    # Assert stored result and full_report
    stored = session.results_store[result["result_id"]]
    record = stored["record"]
    assert record.tool_name == "spearman_filter_tool"

    full_report = stored["full_report"]
    assert full_report is not None
    assert list(full_report.columns) == ["term", "spearman_r"]

    # Check that full_report is sorted descending by spearman_r
    r_in_report = full_report["spearman_r"].tolist()
    assert r_in_report == sorted(r_in_report, reverse=True)

    # Check exact Spearman r values per term
    report_map = dict(zip(full_report["term"], full_report["spearman_r"]))
    for term, exp_val in expected_r.items():
        assert report_map[term] == pytest.approx(exp_val, abs=1e-3)

    # Assert session feature_matrix and feature_names remain unchanged
    assert np.array_equal(session.feature_matrix.toarray(), orig_matrix.toarray())
    assert session.feature_names == orig_feature_names
