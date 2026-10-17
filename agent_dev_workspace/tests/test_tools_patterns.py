import json
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_patterns import make_tools as make_patterns_tools


def setup_fixture_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    return session


def assert_session_preserved(session, df_snap, mat_snap, names_snap, labels_snap, cats_snap):
    assert_frame_equal(session.dataframe, df_snap)
    if mat_snap is None:
        assert session.feature_matrix is None
    else:
        np.testing.assert_array_equal(session.feature_matrix.toarray(), mat_snap)
    assert session.feature_names == names_snap
    np.testing.assert_array_equal(session.labels, labels_snap)
    assert session.categories == cats_snap


def assert_tool_record_and_json(session, res, tool_name, expected_args):
    # Verify JSON serializability with allow_nan=False
    json.dumps(res, allow_nan=False)

    # Verify results_store structure
    res_id = res["result_id"]
    assert res_id in session.results_store
    record = session.results_store[res_id]["record"]
    assert record.tool_name == tool_name
    assert record.args == expected_args
    assert record.summary == res


def test_mine_patterns_zero_surviving_terms_fixture_17():
    session = setup_fixture_session()
    tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    df_snap = session.dataframe.copy(deep=True)
    mat_snap = session.feature_matrix.toarray().copy()
    names_snap = list(session.feature_names)
    labels_snap = session.labels.copy()
    cats_snap = list(session.categories)

    for cat in ["catA", "catB"]:
        for method in ["variance", "term_frequency"]:
            args = {
                "category_name": cat,
                "filtering_method": method,
                "algorithm": "fpgrowth",
                "min_sup": 1,
                "k": None,
            }
            res = mine_patterns_tool.invoke(args)

            assert res["patterns"] == []
            assert "note" in res
            
            art_key = f"patterns_{cat}_{method}_fpgrowth"
            assert art_key in session.artifacts
            assert session.artifacts[art_key] == []

            assert_tool_record_and_json(session, res, "mine_patterns_tool", args)

    assert_session_preserved(session, df_snap, mat_snap, names_snap, labels_snap, cats_snap)


def test_mine_patterns_tfidf_fpgrowth_fixture_17():
    session = setup_fixture_session()
    tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    df_snap = session.dataframe.copy(deep=True)
    mat_snap = session.feature_matrix.toarray().copy()
    names_snap = list(session.feature_names)
    labels_snap = session.labels.copy()
    cats_snap = list(session.categories)

    # catA at min_sup=1 to 3 -> alpha: 3
    for min_sup in [1, 2, 3]:
        args = {
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": min_sup,
            "k": None,
        }
        res_a = mine_patterns_tool.invoke(args)
        assert len(res_a["patterns"]) == 1
        assert res_a["patterns"][0]["pattern"] == ["alpha"]
        assert res_a["patterns"][0]["support"] == 3
        
        art_key = f"patterns_catA_tfidf_fpgrowth"
        assert session.artifacts[art_key] == res_a["patterns"]
        assert_tool_record_and_json(session, res_a, "mine_patterns_tool", args)

    # catA at min_sup=4 -> 0 patterns
    args_a_4 = {
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4,
        "k": None,
    }
    res_a_4 = mine_patterns_tool.invoke(args_a_4)
    assert res_a_4["patterns"] == []
    assert session.artifacts["patterns_catA_tfidf_fpgrowth"] == []
    assert_tool_record_and_json(session, res_a_4, "mine_patterns_tool", args_a_4)

    # catB at min_sup=1 to 4 -> gamma: 4
    for min_sup in [1, 2, 3, 4]:
        args = {
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": min_sup,
            "k": None,
        }
        res_b = mine_patterns_tool.invoke(args)
        assert len(res_b["patterns"]) == 1
        assert res_b["patterns"][0]["pattern"] == ["gamma"]
        assert res_b["patterns"][0]["support"] == 4

        art_key = f"patterns_catB_tfidf_fpgrowth"
        assert session.artifacts[art_key] == res_b["patterns"]
        assert_tool_record_and_json(session, res_b, "mine_patterns_tool", args)

    assert_session_preserved(session, df_snap, mat_snap, names_snap, labels_snap, cats_snap)


def test_mine_patterns_topk_and_maxfpgrowth_fixture_20():
    session = setup_fixture_session()
    tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    df_snap = session.dataframe.copy(deep=True)
    mat_snap = session.feature_matrix.toarray().copy()
    names_snap = list(session.feature_names)
    labels_snap = session.labels.copy()
    cats_snap = list(session.categories)

    # topk for k=1, 2, 3 -> exactly 1 pattern per cat
    for k_val in [1, 2, 3]:
        args_a = {
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "min_sup": None,
            "k": k_val,
        }
        res_a = mine_patterns_tool.invoke(args_a)
        assert len(res_a["patterns"]) == 1
        assert res_a["patterns"][0]["pattern"] == ["alpha"]
        assert res_a["patterns"][0]["support"] == 3
        assert session.artifacts["patterns_catA_tfidf_topk"] == res_a["patterns"]
        assert_tool_record_and_json(session, res_a, "mine_patterns_tool", args_a)

        args_b = {
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "min_sup": None,
            "k": k_val,
        }
        res_b = mine_patterns_tool.invoke(args_b)
        assert len(res_b["patterns"]) == 1
        assert res_b["patterns"][0]["pattern"] == ["gamma"]
        assert res_b["patterns"][0]["support"] == 4
        assert session.artifacts["patterns_catB_tfidf_topk"] == res_b["patterns"]
        assert_tool_record_and_json(session, res_b, "mine_patterns_tool", args_b)

    # maxfpgrowth -> 0 patterns for both categories
    for cat in ["catA", "catB"]:
        for min_sup in [1, 4]:
            args = {
                "category_name": cat,
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": min_sup,
                "k": None,
            }
            res_max = mine_patterns_tool.invoke(args)
            assert res_max["patterns"] == []
            art_key = f"patterns_{cat}_tfidf_maxfpgrowth"
            assert session.artifacts[art_key] == []
            assert_tool_record_and_json(session, res_max, "mine_patterns_tool", args)

    assert_session_preserved(session, df_snap, mat_snap, names_snap, labels_snap, cats_snap)
