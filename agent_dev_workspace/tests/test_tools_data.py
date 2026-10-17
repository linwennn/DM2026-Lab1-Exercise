import json
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools


def test_check_missing_tool_known_answer_2():
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

    # Snapshot initial dataframe
    original_df = session.dataframe.copy()

    # Initialize data tools
    data_tools = make_data_tools(session)
    check_missing_tool = next(
        t for t in data_tools if t.name == "check_missing_tool"
    )

    # 1. Assert fixture #2 known answer: n_missing=1, missing_indices=[7]
    res_fixture = check_missing_tool.invoke({})
    assert res_fixture["n_missing"] == 1
    assert res_fixture["missing_indices"] == [7]

    # Verify session.dataframe was not mutated by fixture invocation
    pd.testing.assert_frame_equal(session.dataframe, original_df)

    # 2. Check stored ResultRecord in session.results_store
    result_id = res_fixture["result_id"]
    assert result_id in session.results_store
    record = session.results_store[result_id]["record"]
    assert record.tool_name == "check_missing_tool"
    assert record.summary == res_fixture

    # 3. Test null, empty, and whitespace-only text values on custom dataframe
    custom_df = pd.DataFrame(
        {
            "text": ["valid text", None, "", "   ", "another text"],
            "category_name": ["catA", "catA", "catB", "catB", "catA"],
        }
    )
    session.dataframe = custom_df
    custom_df_copy = custom_df.copy()

    res_custom = check_missing_tool.invoke({})
    assert res_custom["n_missing"] == 3
    assert res_custom["missing_indices"] == [1, 2, 3]

    # Verify custom_df was not mutated
    pd.testing.assert_frame_equal(session.dataframe, custom_df_copy)

    # 4. Error testing: missing column
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        check_missing_tool.invoke({"text_column": "nonexistent"})

    # 5. Error testing: missing dataset
    session.dataframe = None
    with pytest.raises(ValueError, match="No dataset loaded"):
        check_missing_tool.invoke({})


def test_check_duplicates_tool_known_answer_3():
    # Instantiate single fresh session state for this test
    session = SessionState()

    # Load load_dataset_tool from premade_tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Load fixture dataset into session (8 rows)
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    original_df = session.dataframe.copy()

    # Initialize data tools
    data_tools = make_data_tools(session)
    check_duplicates_tool = next(
        t for t in data_tools if t.name == "check_duplicates_tool"
    )

    # 1. Default check (drop=False): check_duplicates without mutation
    res_check = check_duplicates_tool.invoke({})
    assert res_check["n_duplicates"] == 1
    assert res_check["duplicate_indices"] == [5, 6]
    assert res_check["n_rows_dropped"] == 0
    assert res_check["n_rows_after"] == 8

    # Verify session.dataframe was not mutated during default check
    pd.testing.assert_frame_equal(session.dataframe, original_df)

    # Verify stored ResultRecord for check
    result_id_check = res_check["result_id"]
    assert result_id_check in session.results_store
    rec_check = session.results_store[result_id_check]["record"]
    assert rec_check.tool_name == "check_duplicates_tool"
    assert rec_check.summary == res_check

    # Expected remaining DataFrame after dropping rows 5 and 6 with keep=False
    expected_df = original_df.iloc[[0, 1, 2, 3, 4, 7]].reset_index(drop=True)

    # 2. Drop duplicates (drop=True): removes both copies (keep=False)
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["n_duplicates"] == 1
    assert res_drop["duplicate_indices"] == [5, 6]
    assert res_drop["n_rows_dropped"] == 2
    assert res_drop["n_rows_after"] == 6
    assert len(session.dataframe) == 6

    # Verify remaining DataFrame exactly matches expected_df using assert_frame_equal
    pd.testing.assert_frame_equal(session.dataframe, expected_df)

    # 3. Repeat call after drop=True finds no duplicates remaining
    res_repeat = check_duplicates_tool.invoke({"drop": False})
    assert res_repeat["n_duplicates"] == 0
    assert res_repeat["duplicate_indices"] == []
    assert res_repeat["n_rows_dropped"] == 0
    assert res_repeat["n_rows_after"] == 6


def test_sample_data_tool_known_answer_13():
    # Instantiate single fresh session state for this test
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Load fixture dataset into session (8 rows)
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    original_df = session.dataframe.copy()

    # Initialize data tools
    data_tools = make_data_tools(session)
    sample_data_tool = next(
        t for t in data_tools if t.name == "sample_data_tool"
    )

    # 1. Assert fixture #13 known answer with n=4, random_state=42
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res["n_requested"] == 4
    assert res["n_sampled"] == 4
    assert res["sample_indices"] == [1, 5, 0, 7]

    # Verify sample_data records match sampled indices in original_df
    expected_sample_df = original_df.loc[[1, 5, 0, 7]]
    expected_records = json.loads(expected_sample_df.to_json(orient="records"))
    assert res["sample_data"] == expected_records

    # 2. Reproducibility test (same n and seed yields identical output)
    res_repeat = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res_repeat["sample_indices"] == res["sample_indices"]
    assert res_repeat["sample_data"] == res["sample_data"]

    # 3. JSON serializability assertion
    serialized_json = json.dumps(res, allow_nan=False)
    assert isinstance(serialized_json, str)

    # 4. Verify stored ResultRecord in session.results_store
    result_id = res["result_id"]
    assert result_id in session.results_store
    record = session.results_store[result_id]["record"]
    assert record.tool_name == "sample_data_tool"
    assert record.args == {"n": 4, "random_state": 42}
    assert record.summary == res

    # 5. Verify session.dataframe was NOT mutated
    pd.testing.assert_frame_equal(session.dataframe, original_df)


def test_tokenize_tool_known_answer_10():
    # Instantiate single fresh session state for this test
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Load fixture dataset into session (8 rows)
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    original_df = session.dataframe.copy()

    # Initialize data tools
    data_tools = make_data_tools(session)
    tokenize_tool = next(
        t for t in data_tools if t.name == "tokenize_tool"
    )

    # Known answer #10 expected tokens per row
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    # 1. Run tokenize_tool on fixture
    res = tokenize_tool.invoke({})
    assert res["text_column"] == "text"
    assert res["n_documents"] == 8
    assert res["total_tokens"] == 27
    assert res["sample_tokens"] == expected_tokens[:3]

    # 2. Verify all 8 expected token lists match session.dataframe["unigrams"] and session.artifacts["unigrams"]
    assert session.dataframe["unigrams"].tolist() == expected_tokens
    assert session.artifacts["unigrams"] == expected_tokens
    assert session.dataframe["unigrams"].iloc[7] == []

    # 3. Check stored ResultRecord in session.results_store
    result_id = res["result_id"]
    assert result_id in session.results_store
    record = session.results_store[result_id]["record"]
    assert record.tool_name == "tokenize_tool"
    assert record.args == {"text_column": "text"}
    assert record.summary == res

    # 4. Verify original DataFrame columns remain unchanged
    for col in original_df.columns:
        pd.testing.assert_series_equal(session.dataframe[col], original_df[col])

    # 5. Check lowercasing behavior on custom mixed-case text
    custom_df = pd.DataFrame({"text": ["ALWAYS Alpha BETA"]})
    session.dataframe = custom_df
    res_mixed = tokenize_tool.invoke({})
    assert res_mixed["total_tokens"] == 3
    assert session.dataframe["unigrams"].tolist() == [["always", "alpha", "beta"]]


def test_list_files_tool_known_answer_11():
    # Instantiate single fresh session state for this test
    session = SessionState()

    # Initialize data tools
    data_tools = make_data_tools(session)
    list_files_tool = next(
        t for t in data_tools if t.name == "list_files_tool"
    )

    # 1. Known answer #11: subdirectory="newdataset"
    res = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert res["subdirectory"] == "newdataset"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]

    # 2. Verify stored ResultRecord in session.results_store
    result_id = res["result_id"]
    assert result_id in session.results_store
    record = session.results_store[result_id]["record"]
    assert record.tool_name == "list_files_tool"
    assert record.args == {"subdirectory": "newdataset"}
    assert record.summary == res

    # 3. Error testing for missing/non-existent directory
    with pytest.raises(ValueError, match="Directory 'nonexistent_dir' does not exist"):
        list_files_tool.invoke({"subdirectory": "nonexistent_dir"})


def test_describe_data_tool_known_answer_18():
    # Instantiate single fresh session state for this test
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Load fixture dataset into session (8 rows)
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    original_df = session.dataframe.copy()

    # Initialize data tools
    data_tools = make_data_tools(session)
    describe_data_tool = next(
        t for t in data_tools if t.name == "describe_data_tool"
    )

    try:
        # 1. Known answer #18 assertions
        res = describe_data_tool.invoke({})
        overall = res["overall_stats"]
        assert overall["count"] == pytest.approx(8.0)
        assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
        assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
        assert overall["min"] == pytest.approx(0.0)
        assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
        assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
        assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
        assert overall["max"] == pytest.approx(30.0)

        per_cat = res["per_category_stats"]
        catA = per_cat["catA"]
        assert catA["count"] == pytest.approx(4.0)
        assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
        assert catA["std"] == pytest.approx(12.5, abs=1e-3)
        assert catA["min"] == pytest.approx(0.0)
        assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
        assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
        assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
        assert catA["max"] == pytest.approx(29.0)

        catB = per_cat["catB"]
        assert catB["count"] == pytest.approx(4.0)
        assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
        assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
        assert catB["min"] == pytest.approx(18.0)
        assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
        assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
        assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
        assert catB["max"] == pytest.approx(30.0)

        # 2. Check pending figure set
        assert session.pending_figure is not None
        assert isinstance(session.pending_figure, plt.Figure)

        # 3. Check stored ResultRecord in session.results_store
        result_id = res["result_id"]
        assert result_id in session.results_store
        record = session.results_store[result_id]["record"]
        assert record.tool_name == "describe_data_tool"
        assert record.summary == res

        # 4. Verify session.dataframe was NOT mutated
        pd.testing.assert_frame_equal(session.dataframe, original_df)

        # 5. Test singleton category case (1 row) -> std is None, and json.dumps works without NaN
        singleton_df = pd.DataFrame({"text": ["hello"], "category_name": ["catSingle"]})
        session.dataframe = singleton_df
        res_single = describe_data_tool.invoke({})
        assert res_single["per_category_stats"]["catSingle"]["std"] is None

        json_str = json.dumps(res_single, allow_nan=False)
        assert isinstance(json_str, str)
    finally:
        plt.close("all")


def test_inspect_data_tool():
    # Instantiate single fresh session state for this test
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Load fixture dataset into session (8 rows)
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    original_df = session.dataframe.copy()

    # Initialize data tools
    data_tools = make_data_tools(session)
    inspect_data_tool = next(
        t for t in data_tools if t.name == "inspect_data_tool"
    )

    # 1. Default invocation (n=5)
    res_default = inspect_data_tool.invoke({})
    assert res_default["n_rows"] == 8
    assert res_default["n_columns"] == len(original_df.columns)
    assert res_default["columns"] == list(original_df.columns)
    assert res_default["n_requested"] == 5
    expected_head_5 = json.loads(original_df.head(5).to_json(orient="records"))
    assert res_default["head_records"] == expected_head_5

    # Check stored ResultRecord in session.results_store
    result_id = res_default["result_id"]
    assert result_id in session.results_store
    record = session.results_store[result_id]["record"]
    assert record.tool_name == "inspect_data_tool"
    assert record.summary == res_default

    # Strict JSON serialization test
    json_str = json.dumps(res_default, allow_nan=False)
    assert isinstance(json_str, str)

    # 2. Specific n (n=2)
    res_2 = inspect_data_tool.invoke({"n": 2})
    assert res_2["n_requested"] == 2
    assert len(res_2["head_records"]) == 2
    expected_head_2 = json.loads(original_df.head(2).to_json(orient="records"))
    assert res_2["head_records"] == expected_head_2

    # 3. n larger than dataset (n=20 on 8-row df)
    res_large = inspect_data_tool.invoke({"n": 20})
    assert res_large["n_requested"] == 20
    assert len(res_large["head_records"]) == 8
    expected_head_all = json.loads(original_df.head(20).to_json(orient="records"))
    assert res_large["head_records"] == expected_head_all

    # 4. Verify session.dataframe was NOT mutated
    pd.testing.assert_frame_equal(session.dataframe, original_df)

    # 5. Error handling: n <= 0
    with pytest.raises(ValueError, match="Parameter n must be greater than 0"):
        inspect_data_tool.invoke({"n": 0})

    with pytest.raises(ValueError, match="Parameter n must be greater than 0"):
        inspect_data_tool.invoke({"n": -3})

    # 6. Error handling: missing dataset
    session.dataframe = None
    with pytest.raises(ValueError, match="No dataset loaded"):
        inspect_data_tool.invoke({})
