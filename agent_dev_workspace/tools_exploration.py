from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc_1: Union[int, str],
        doc_2: Union[int, str],
    ) -> dict:
        """Computes cosine similarity between two documents specified by index or raw text.

        Args:
            doc_1: Document index (int) in session.feature_matrix or raw document text (str).
            doc_2: Document index (int) in session.feature_matrix or raw document text (str).

        Returns:
            A summary dictionary containing result_id, doc_1, doc_2, and similarity score.
        """
        def get_vector(doc_input):
            if isinstance(doc_input, int):
                if session.feature_matrix is None:
                    raise ValueError(
                        "No document-term matrix found in session. Run build_dtm_tool first."
                    )
                n_docs = session.feature_matrix.shape[0]
                if not (0 <= doc_input < n_docs):
                    raise ValueError(
                        f"Document index {doc_input} out of bounds [0, {n_docs - 1}]."
                    )
                return session.feature_matrix[doc_input].toarray().ravel()
            elif isinstance(doc_input, str):
                vectorizer = session.artifacts.get("count_vectorizer")
                if vectorizer is None:
                    raise ValueError(
                        "No count vectorizer found in session.artifacts. Run build_dtm_tool first."
                    )
                return vectorizer.transform([doc_input]).toarray().ravel()
            else:
                raise ValueError("Document input must be an int (index) or a str (text).")

        vec1 = get_vector(doc_1)
        vec2 = get_vector(doc_2)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_1": doc_1,
            "doc_2": doc_2,
            "similarity": similarity,
        }

        session.store_result(
            tool_name="cosine_similarity_tool",
            args={"doc_1": doc_1, "doc_2": doc_2},
            summary=summary,
        )

        return summary

    @tool
    def feature_correlation_matrix_tool(top_k: int = 20) -> dict:
        """Computes Pearson correlation matrix over top-k terms by variance and renders a heatmap.

        Args:
            top_k: Maximum number of top terms by variance to compute correlation for (default 20).

        Returns:
            A summary dictionary containing result_id, top_terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "No document-term matrix found in session. Run build_dtm_tool first."
            )

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        # Compute column variances sparsely: var(X) = E[X^2] - (E[X])^2
        X = session.feature_matrix
        means = np.asarray(X.mean(axis=0)).ravel()
        mean_sqs = np.asarray(X.power(2).mean(axis=0)).ravel()
        variances = mean_sqs - (means ** 2)

        # Stable descending sort for top indices: np.argsort(-variances, kind="stable")
        sorted_indices = np.argsort(-variances, kind="stable")
        k = min(top_k, len(session.feature_names))
        top_indices = sorted_indices[:k]

        top_terms = [session.feature_names[i] for i in top_indices]

        # Densify ONLY the selected top_indices columns
        sub_matrix_dense = X[:, top_indices].toarray()

        # Compute Pearson correlation matrix across variables (terms as columns -> rowvar=False)
        corr_matrix = np.corrcoef(sub_matrix_dense, rowvar=False)
        if np.ndim(corr_matrix) == 0:
            corr_matrix = np.array([[float(corr_matrix)]])

        # Render heatmap
        fig, ax = plt.subplots(figsize=(max(6, k * 0.8), max(5, k * 0.7)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
            vmin=-1.0,
            vmax=1.0,
        )
        ax.set_title(f"Feature Correlation Matrix (Top-{k} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        corr_list = corr_matrix.tolist()
        result_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": result_id,
            "top_terms": top_terms,
            "correlation_matrix": corr_list,
        }

        session.store_result(
            tool_name="feature_correlation_matrix_tool",
            args={"top_k": top_k},
            summary=summary,
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
