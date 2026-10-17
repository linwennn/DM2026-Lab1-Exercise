from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ngram_range: List[int] = [1, 1],
    ) -> dict:
        """Builds a document-term matrix from session.dataframe['text'].

        Args:
            max_features: Maximum number of terms to keep by term frequency.
            min_df: Minimum document frequency cutoff (count or proportion).
            max_df: Maximum document frequency cutoff (count or proportion).
            ngram_range: List of two integers [min_n, max_n] specifying n-gram range.

        Returns:
            A summary dictionary containing result_id, status, document/term counts,
            vocabulary preview (first 50 terms in column order), and matrix sparsity metrics.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError(
                "No dataset loaded or 'text' column missing in session.dataframe."
            )

        # Validate ngram_range
        if (
            not isinstance(ngram_range, (list, tuple))
            or len(ngram_range) != 2
            or not all(isinstance(x, int) for x in ngram_range)
            or ngram_range[0] < 1
            or ngram_range[1] < ngram_range[0]
        ):
            raise ValueError(
                f"Invalid ngram_range: {ngram_range}. Must be a list of two positive integers [min_n, max_n] with min_n <= max_n."
            )

        ngram_tuple = tuple(ngram_range)

        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
        )

        texts = session.dataframe["text"].fillna("").astype(str)
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        n_documents, n_terms = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_documents * n_terms
        sparsity_pct = (
            100.0 * (1.0 - (non_zero / total_elements))
            if total_elements > 0
            else 0.0
        )

        vocab_preview = feature_names[:50]

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "status": "Document-Term Matrix built successfully.",
            "n_documents": n_documents,
            "n_terms": n_terms,
            "vocabulary": vocab_preview,
            "vocabulary_preview_size": len(vocab_preview),
            "truncated": n_terms > 50,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct,
        }

        session.store_result(
            tool_name="build_dtm_tool",
            args={
                "max_features": max_features,
                "min_df": min_df,
                "max_df": max_df,
                "ngram_range": ngram_range,
            },
            summary=summary,
        )

        return summary

    @tool
    def term_frequency_tool() -> dict:
        """Aggregates term frequencies across all documents in session.feature_matrix.

        Returns:
            A summary dictionary containing result_id, n_terms, total_word_count,
            and a mapping of top term frequencies sorted descending (up to 50 terms).
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "No document-term matrix found in session. Build DTM using build_dtm_tool first."
            )

        # Sum matrix columns across all documents
        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        term_freqs = {
            term: int(count)
            for term, count in zip(session.feature_names, counts)
        }
        # Sort descending by frequency
        sorted_freqs = dict(
            sorted(term_freqs.items(), key=lambda item: item[1], reverse=True)
        )

        total_word_count = int(counts.sum())
        n_terms = len(session.feature_names)

        top_freqs = dict(list(sorted_freqs.items())[:50])

        full_report_df = pd.DataFrame(
            list(sorted_freqs.items()), columns=["term", "frequency"]
        )

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "n_terms": n_terms,
            "total_word_count": total_word_count,
            "term_frequencies": top_freqs,
            "term_frequencies_preview_size": len(top_freqs),
            "truncated": n_terms > 50,
        }

        session.store_result(
            tool_name="term_frequency_tool",
            args={},
            summary=summary,
            full_report=full_report_df,
        )

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> dict:
        """Renders an annotated heatmap of a positional slice of the raw document-term matrix.

        Args:
            n_terms: Number of terms (columns) to slice from the DTM in vocabulary order. Default 20.
            n_documents: Number of documents (rows) to slice from the DTM in document order. Default 20.

        Returns:
            A summary dictionary containing result_id, n_terms, n_documents,
            selected_terms, selected_documents, and matrix_slice (nested list).
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "No document-term matrix found in session. Build DTM using build_dtm_tool first."
            )

        if n_terms <= 0 or n_documents <= 0:
            raise ValueError("Both n_terms and n_documents must be positive integers.")

        total_docs, total_terms = session.feature_matrix.shape
        actual_docs = min(n_documents, total_docs)
        actual_terms = min(n_terms, total_terms)

        slice_matrix = session.feature_matrix[0:actual_docs, 0:actual_terms].toarray()
        selected_terms = session.feature_names[0:actual_terms]
        selected_documents = [f"Doc {i}" for i in range(actual_docs)]

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.8), max(4, actual_docs * 0.5)))
        sns.heatmap(
            slice_matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=selected_terms,
            yticklabels=selected_documents,
            ax=ax,
        )
        ax.set_title("Document-Term Matrix Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_terms": actual_terms,
            "n_documents": actual_docs,
            "selected_terms": selected_terms,
            "selected_documents": selected_documents,
            "matrix_slice": slice_matrix.tolist(),
        }

        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary,
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
