import matplotlib.pyplot as plt
import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    # Instantiate single session state
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

    # Initialize student tool from tools_dtm
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Invoke build_dtm_tool with default parameters
    result = build_dtm_tool.invoke({})

    # Expected vocabulary according to known answer #4
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]

    # Expected 8x5 DTM matrix values (from TEST_FIXTURE.md #5 and #21)
    # Columns correspond to: ["alpha", "always", "beta", "delta", "gamma"]
    expected_matrix = np.array(
        [
            [3, 1, 1, 0, 0],  # Doc 0
            [2, 1, 1, 0, 0],  # Doc 1
            [1, 1, 1, 0, 0],  # Doc 2
            [0, 1, 0, 1, 3],  # Doc 3
            [0, 1, 0, 1, 2],  # Doc 4
            [0, 1, 0, 1, 1],  # Doc 5
            [0, 1, 0, 1, 1],  # Doc 6
            [0, 0, 0, 0, 0],  # Doc 7 (empty text)
        ]
    )

    # Assert returned summary values
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["vocabulary"] == expected_vocab
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Assert session feature_matrix and feature_names
    assert session.feature_matrix is not None
    assert session.feature_names == expected_vocab
    assert np.array_equal(session.feature_matrix.toarray(), expected_matrix)
    assert "count_vectorizer" in session.artifacts

    # Assert session labels and categories
    expected_labels = np.array(
        ["catA", "catA", "catA", "catB", "catB", "catB", "catB", "catA"]
    )
    assert np.array_equal(session.labels, expected_labels)
    assert session.categories == ["catA", "catB"]

    # Assert session.results_store record
    result_id = result["result_id"]
    assert result_id in session.results_store
    stored_record = session.results_store[result_id]
    record = stored_record["record"]
    assert record.tool_name == "build_dtm_tool"
    assert record.summary == result


def test_term_frequency_tool_known_answer_14():
    # Instantiate single session state for this test
    session = SessionState()

    # Load load_dataset_tool from premade_tools
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

    # Initialize student tools from tools_dtm
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(
        t for t in dtm_tools if t.name == "term_frequency_tool"
    )

    # Build DTM first
    build_dtm_tool.invoke({})

    # Preserve snapshot of DTM state before term_frequency_tool execution
    original_matrix = session.feature_matrix.toarray().copy()
    original_names = list(session.feature_names)

    # Invoke term_frequency_tool
    result = term_frequency_tool.invoke({})

    # Assert expected counts according to TEST_FIXTURE.md #14
    assert result["n_terms"] == 5
    assert result["total_word_count"] == 27
    assert result["term_frequencies"] == {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3,
    }
    # Verify dictionary is in descending frequency order
    freq_values = list(result["term_frequencies"].values())
    assert freq_values == sorted(freq_values, reverse=True)

    # Assert stored result in session.results_store
    result_id = result["result_id"]
    assert result_id in session.results_store
    record = session.results_store[result_id]["record"]
    assert record.tool_name == "term_frequency_tool"
    assert record.summary == result

    # Assert session feature_matrix and feature_names were not modified
    assert np.array_equal(session.feature_matrix.toarray(), original_matrix)
    assert session.feature_names == original_names


def test_dtm_heatmap_tool_known_answer_21():
    # Instantiate single session state
    session = SessionState()

    # Load load_dataset_tool from premade_tools
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

    # Initialize student tools from tools_dtm
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(
        t for t in dtm_tools if t.name == "dtm_heatmap_tool"
    )

    # Build DTM first
    build_dtm_tool.invoke({})

    original_matrix = session.feature_matrix.toarray().copy()
    original_names = list(session.feature_names)

    fig1 = None
    fig2 = None

    try:
        # Test 1: Default call (capped at 8 docs x 5 terms)
        result_default = dtm_heatmap_tool.invoke({})

        expected_full_slice = [
            [3, 1, 1, 0, 0],
            [2, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 0, 1, 3],
            [0, 1, 0, 1, 2],
            [0, 1, 0, 1, 1],
            [0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
        ]

        assert result_default["n_documents"] == 8
        assert result_default["n_terms"] == 5
        assert result_default["selected_terms"] == [
            "alpha",
            "always",
            "beta",
            "delta",
            "gamma",
        ]
        assert result_default["selected_documents"] == [f"Doc {i}" for i in range(8)]
        assert result_default["matrix_slice"] == expected_full_slice

        # Check figure 1 and cell annotations (8x5 = 40 annotated cells)
        fig1 = session.pending_figure
        assert fig1 is not None
        assert isinstance(fig1, plt.Figure)
        ax1 = fig1.axes[0]
        assert len(ax1.texts) == 40

        # Test 2: Positional slice (n_documents=3, n_terms=2)
        result_slice = dtm_heatmap_tool.invoke({"n_documents": 3, "n_terms": 2})

        expected_3x2_slice = [
            [3, 1],
            [2, 1],
            [1, 1],
        ]

        assert result_slice["n_documents"] == 3
        assert result_slice["n_terms"] == 2
        assert result_slice["selected_terms"] == ["alpha", "always"]
        assert result_slice["selected_documents"] == ["Doc 0", "Doc 1", "Doc 2"]
        assert result_slice["matrix_slice"] == expected_3x2_slice

        # Check figure 2, ensuring it's a distinct figure object with 3x2 = 6 annotated cells
        fig2 = session.pending_figure
        assert fig2 is not None
        assert isinstance(fig2, plt.Figure)
        assert fig2 is not fig1
        ax2 = fig2.axes[0]
        assert len(ax2.texts) == 6

        # Assert original session feature_matrix and feature_names were preserved
        assert np.array_equal(session.feature_matrix.toarray(), original_matrix)
        assert session.feature_names == original_names
    finally:
        if fig1 is not None:
            plt.close(fig1)
        if fig2 is not None:
            plt.close(fig2)
