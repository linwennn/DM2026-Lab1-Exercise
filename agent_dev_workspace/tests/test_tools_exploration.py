import matplotlib.pyplot as plt
import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_tool_known_answer_16():
    # Instantiate single shared session state
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Load fixture dataset into session
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Initialize DTM tools and build document-term matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Snapshot DTM and feature names
    original_matrix = session.feature_matrix.toarray().copy()
    original_names = list(session.feature_names)

    # Initialize exploration tools
    exploration_tools = make_exploration_tools(session)
    cosine_tool = next(
        t for t in exploration_tools if t.name == "cosine_similarity_tool"
    )

    # 1. Assert four known answer similarities from TEST_FIXTURE.md #16 by index
    # Doc 0 vs 1 (both catA, near-identical): 0.9847
    res_0_1 = cosine_tool.invoke({"doc_1": 0, "doc_2": 1})
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-4)

    # Doc 0 vs 3 (catA vs catB, share only "always"): 0.0909
    res_0_3 = cosine_tool.invoke({"doc_1": 0, "doc_2": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-4)

    # Doc 3 vs 4 (both catB, near-identical): 0.9847
    res_3_4 = cosine_tool.invoke({"doc_1": 3, "doc_2": 4})
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-4)

    # Doc 0 vs 7 (catA vs empty text row): 0.0
    res_0_7 = cosine_tool.invoke({"doc_1": 0, "doc_2": 7})
    assert res_0_7["similarity"] == 0.0

    # 2. Equivalent raw text inputs
    text_0 = session.dataframe.loc[0, "text"]
    text_1 = session.dataframe.loc[1, "text"]
    res_text = cosine_tool.invoke({"doc_1": text_0, "doc_2": text_1})
    assert res_text["similarity"] == pytest.approx(0.9847, abs=1e-4)

    # 3. Mixed input (index and raw text)
    res_mixed = cosine_tool.invoke({"doc_1": 0, "doc_2": text_1})
    assert res_mixed["similarity"] == pytest.approx(0.9847, abs=1e-4)

    # 4. Zero vector string (text with no vocabulary terms)
    res_empty_str = cosine_tool.invoke({"doc_1": 0, "doc_2": "nonexistent term"})
    assert res_empty_str["similarity"] == 0.0

    # 5. Invalid document indices raise ValueError
    with pytest.raises(ValueError):
        cosine_tool.invoke({"doc_1": -1, "doc_2": 0})

    with pytest.raises(ValueError):
        cosine_tool.invoke({"doc_1": 0, "doc_2": 100})

    # 6. Verify stored ResultRecord in session.results_store
    result_id = res_0_1["result_id"]
    assert result_id in session.results_store
    record = session.results_store[result_id]["record"]
    assert record.tool_name == "cosine_similarity_tool"
    assert record.summary == res_0_1

    # 7. Confirm DTM and feature names remain unchanged
    assert np.array_equal(session.feature_matrix.toarray(), original_matrix)
    assert session.feature_names == original_names


def test_feature_correlation_matrix_tool_known_answer_19():
    # Instantiate single session state
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Load fixture dataset into session
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Snapshot DTM and feature names
    original_matrix = session.feature_matrix.toarray().copy()
    original_names = list(session.feature_names)

    # Initialize exploration tools
    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(
        t for t in exploration_tools
        if t.name == "feature_correlation_matrix_tool"
    )

    fig = None
    try:
        # Invoke tool with default top_k=20
        result = feature_corr_tool.invoke({})

        # 1. Expected term order by variance descending (TEST_FIXTURE.md #19)
        expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
        assert result["top_terms"] == expected_terms

        # 2. Expected 5x5 Pearson correlation matrix (TEST_FIXTURE.md #19)
        expected_corr = np.array(
            [
                [1.0000, -0.5718, -0.6882, 0.8885, 0.2601],
                [-0.5718, 1.0000, 0.8307, -0.6435, 0.3140],
                [-0.6882, 0.8307, 1.0000, -0.7746, 0.3780],
                [0.8885, -0.6435, -0.7746, 1.0000, 0.2928],
                [0.2601, 0.3140, 0.3780, 0.2928, 1.0000],
            ]
        )

        matrix_arr = np.array(result["correlation_matrix"])
        assert matrix_arr.shape == (5, 5)
        assert matrix_arr == pytest.approx(expected_corr, abs=1e-4)

        # 3. Check session.pending_figure and annotations (5x5 = 25 cells)
        fig = session.pending_figure
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.texts) == 25

        # 4. Check stored ResultRecord in session.results_store
        result_id = result["result_id"]
        assert result_id in session.results_store
        record = session.results_store[result_id]["record"]
        assert record.tool_name == "feature_correlation_matrix_tool"
        assert record.summary == result

        # 5. Confirm DTM and feature names remain unchanged
        assert np.array_equal(session.feature_matrix.toarray(), original_matrix)
        assert session.feature_names == original_names
    finally:
        if fig is not None:
            plt.close(fig)
