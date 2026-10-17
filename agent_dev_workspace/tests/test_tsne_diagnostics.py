import numpy as np
import pytest
from sklearn.manifold import TSNE

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools


@pytest.mark.skip(reason="Intentional failure for completed diagnostics – keep fixture #8 unchanged")
def test_tsne_diagnostics():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    X_orig = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)

    expected_coords = np.array([
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973]
    ])

    results = {}
    dtypes_to_test = [
        ("original", X_orig),
        ("float64", X_orig.astype(np.float64)),
        ("float32", X_orig.astype(np.float32)),
    ]

    for label, X_mat in dtypes_to_test:
        tsne = TSNE(
            n_components=2,
            perplexity=2,
            random_state=42
        )
        coords = tsne.fit_transform(X_mat)
        max_err = float(np.max(np.abs(coords - expected_coords)))
        results[f"dtype={label}({X_mat.dtype})"] = round(max_err, 4)

    raise AssertionError(f"TSNE Fixture #8 Max Abs Errors across input dtypes: {results}")
