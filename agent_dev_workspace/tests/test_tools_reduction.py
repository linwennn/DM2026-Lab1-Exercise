import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools as make_reduction_tools


def test_binarize_labels_tool():
    # Instantiate single fresh session state for this test
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Load fixture dataset into session (8 rows: 4 catA, 4 catB)
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Set session.labels explicitly from category_name
    session.set_labels(session.dataframe["category_name"].values)

    original_df = session.dataframe.copy()
    original_labels = session.labels.copy()
    original_categories = list(session.categories)

    # Initialize reduction tools
    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(
        t for t in reduction_tools if t.name == "binarize_labels_tool"
    )

    # Expected binarized labels for catB on sample_fixture.csv (catA rows 0,1,2,7 -> 0; catB rows 3,4,5,6 -> 1)
    expected_binary_catB = [0, 0, 0, 1, 1, 1, 1, 0]

    # 1. Run binarize_labels_tool for catB
    res_catB = binarize_labels_tool.invoke({"target_category": "catB"})
    assert res_catB["target_category"] == "catB"
    assert res_catB["positive_count"] == 4
    assert res_catB["negative_count"] == 4
    assert res_catB["binarized_labels"] == expected_binary_catB

    # Check artifacts set correctly
    assert session.artifacts["binarized_labels"] == expected_binary_catB
    assert session.artifacts["binarized_target_category"] == "catB"

    # Check stored ResultRecord in session.results_store
    result_id = res_catB["result_id"]
    assert result_id in session.results_store
    record = session.results_store[result_id]["record"]
    assert record.tool_name == "binarize_labels_tool"
    assert record.args == {"target_category": "catB"}
    assert record.summary == res_catB

    # Strict JSON serialization check
    json_str = json.dumps(res_catB, allow_nan=False)
    assert isinstance(json_str, str)

    # Verify original DataFrame, labels, and categories are preserved
    pd.testing.assert_frame_equal(session.dataframe, original_df)
    np.testing.assert_array_equal(session.labels, original_labels)
    assert session.categories == original_categories

    # 2. Test stale session.categories verification (labels are authoritative)
    session.categories = ["stale1", "stale2"]
    res_catA = binarize_labels_tool.invoke({"target_category": "catA"})
    assert res_catA["positive_count"] == 4
    assert res_catA["negative_count"] == 4
    assert res_catA["binarized_labels"] == [1, 1, 1, 0, 0, 0, 0, 1]

    # 3. Error testing: invalid target category
    with pytest.raises(ValueError, match="Target category 'invalid_cat' not found"):
        binarize_labels_tool.invoke({"target_category": "invalid_cat"})

    # 4. Error testing: row-count mismatch
    session.labels = np.array(["catA", "catB"])  # Only 2 labels vs 8 rows in dataframe
    with pytest.raises(ValueError, match="Length mismatch"):
        binarize_labels_tool.invoke({"target_category": "catA"})

    # 5. Error testing: missing labels
    session.labels = None
    with pytest.raises(ValueError, match="No session.labels available"):
        binarize_labels_tool.invoke({"target_category": "catA"})


def _setup_reduction_session():
    """Helper to set up SessionState with sample_fixture.csv and build DTM via build_dtm_tool."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    return session


def test_reduce_dimensions_pca_known_answer_7():
    session = _setup_reduction_session()
    original_fm = session.feature_matrix.copy()
    original_labels = session.labels.copy()

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    try:
        res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2, "random_state": 42})
        assert res["method"] == "pca"
        assert res["n_components"] == 2

        coords = np.array(res["coordinates"])
        assert coords.shape == (8, 2)
        assert np.all(np.isfinite(coords))

        expected_coords = np.array([
            [2.3669, 0.8871],
            [1.7039, 0.2642],
            [1.0408, -0.3587],
            [-2.0759, 1.0129],
            [-1.4554, 0.3405],
            [-0.8349, -0.3319],
            [-0.8349, -0.3319],
            [0.0894, -1.4822],
        ])

        # Tolerance 1e-3 matching TEST_FIXTURE.md recommendation
        np.testing.assert_allclose(coords, expected_coords, atol=1e-3)

        expected_evr = [0.7532, 0.1965]
        assert res["explained_variance_ratio"] == pytest.approx(expected_evr, abs=1e-3)

        # Check category scatter plot figure and legend
        fig = session.pending_figure
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "catA" in legend_texts and "catB" in legend_texts

        # Check stored ResultRecord and artifacts
        result_id = res["result_id"]
        assert result_id in session.results_store
        assert session.results_store[result_id]["record"].summary == res
        assert session.artifacts["reduced_coordinates"] == res["coordinates"]

        # Strict JSON test
        assert isinstance(json.dumps(res, allow_nan=False), str)

        # Verify DTM and labels non-mutation
        np.testing.assert_array_equal(session.feature_matrix.toarray(), original_fm.toarray())
        np.testing.assert_array_equal(session.labels, original_labels)
    finally:
        plt.close("all")


def test_reduce_dimensions_tsne_known_answer_8():
    session = _setup_reduction_session()
    original_fm = session.feature_matrix.copy()
    original_labels = session.labels.copy()

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    try:
        res = reduce_dimensions_tool.invoke({
            "method": "tsne",
            "perplexity": 2,
            "random_state": 42,
        })
        assert res["method"] == "tsne"

        coords = np.array(res["coordinates"])
        assert coords.shape == (8, 2)
        assert np.all(np.isfinite(coords))

        expected_coords = np.array([
            [1330.9991, 32.1039],
            [951.2459, 393.5588],
            [530.8209, 682.9791],
            [-1558.6448, -314.1625],
            [-915.5153, -346.3907],
            [-280.0313, -1261.2115],
            [-280.0313, -1261.2115],
            [-701.3674, 1465.3973],
        ])

        np.testing.assert_allclose(coords, expected_coords, atol=1e-3)

        fig = session.pending_figure
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "catA" in legend_texts and "catB" in legend_texts

        result_id = res["result_id"]
        assert result_id in session.results_store
        assert session.results_store[result_id]["record"].summary == res
        assert session.artifacts["reduced_coordinates"] == res["coordinates"]

        np.testing.assert_array_equal(session.feature_matrix.toarray(), original_fm.toarray())
        np.testing.assert_array_equal(session.labels, original_labels)
    finally:
        plt.close("all")


def test_reduce_dimensions_umap_known_answer_9():
    session = _setup_reduction_session()
    original_fm = session.feature_matrix.copy()
    original_labels = session.labels.copy()

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    try:
        res = reduce_dimensions_tool.invoke({
            "method": "umap",
            "n_neighbors": 3,
            "random_state": 42,
        })
        assert res["method"] == "umap"

        coords = np.array(res["coordinates"])
        assert coords.shape == (8, 2)
        assert np.all(np.isfinite(coords))

        expected_coords = np.array([
            [6.6811, -6.8218],
            [6.9907, -7.2245],
            [7.7022, -7.3712],
            [10.2228, -6.2310],
            [9.8759, -5.6537],
            [9.3831, -6.5748],
            [9.2153, -5.8297],
            [8.5860, -7.2622],
        ])

        np.testing.assert_allclose(coords, expected_coords, atol=1e-3)

        fig = session.pending_figure
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "catA" in legend_texts and "catB" in legend_texts

        result_id = res["result_id"]
        assert result_id in session.results_store
        assert session.results_store[result_id]["record"].summary == res
        assert session.artifacts["reduced_coordinates"] == res["coordinates"]

        np.testing.assert_array_equal(session.feature_matrix.toarray(), original_fm.toarray())
        np.testing.assert_array_equal(session.labels, original_labels)
    finally:
        plt.close("all")
