import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Flags near-constant terms by variance for removal from the DTM.

        Args:
            threshold: Terms with variance less than or equal to this threshold
                are flagged for removal (default: 0.0).

        Returns:
            Dict containing result_id, threshold, total_terms, retained_terms,
            removed_terms, retained_list, and removed_list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature matrix found. Run build_dtm_tool first.")

        matrix = session.feature_matrix
        feature_names = np.array(session.feature_names)

        # Compute variance per term across documents (ddof=0 for population variance)
        if issparse(matrix):
            # For sparse matrix, Var(X) = E[X^2] - (E[X])^2
            mean = np.ravel(matrix.mean(axis=0))
            sq_mean = np.ravel(matrix.power(2).mean(axis=0))
            variances = sq_mean - (mean ** 2)
        else:
            variances = np.var(matrix, axis=0, ddof=0)

        # Retain terms with variance > threshold, remove terms <= threshold
        retained_mask = variances > threshold
        removed_mask = variances <= threshold

        retained_terms = feature_names[retained_mask].tolist()
        removed_terms = feature_names[removed_mask].tolist()

        # Build full_report DataFrame sorted descending by variance
        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        retained_preview = retained_terms[:20]
        removed_preview = removed_terms[:20]

        result_id = session.next_result_id("variance_filter")
        args = {"threshold": threshold}

        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "retained_terms": len(retained_terms),
            "removed_terms": len(removed_terms),
            "retained_list": retained_preview,
            "removed_list": removed_preview,
            "retained_list_preview_size": len(retained_preview),
            "removed_list_preview_size": len(removed_preview),
            "truncated": len(retained_terms) > 20 or len(removed_terms) > 20,
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args=args,
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Calculates Pearson correlation between term counts and a target class indicator.

        Args:
            target_class: Category name to correlate against.
            threshold: Terms with absolute Pearson correlation less than or equal to
                this threshold are flagged for removal (default: 0.0).

        Returns:
            Dict containing result_id, target_class, threshold, total_terms,
            retained_terms, removed_terms, retained_list, and removed_list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature matrix found. Run build_dtm_tool first.")

        if session.labels is None or len(session.labels) != session.feature_matrix.shape[0]:
            raise ValueError("Session labels missing or length mismatch with feature matrix.")

        if session.categories is None or target_class not in session.categories:
            raise ValueError(f"Invalid target_class '{target_class}'. Options: {session.categories}")

        matrix = session.feature_matrix
        feature_names = np.array(session.feature_names)

        # Binary target indicator vector y (1.0 if label == target_class, else 0.0)
        y = (np.array(session.labels) == target_class).astype(float)

        # Compute Pearson correlation r per column without densifying the whole matrix
        pearson_r_list = []
        n_terms = matrix.shape[1]
        std_y = np.std(y)

        for col_idx in range(n_terms):
            if issparse(matrix):
                col_vec = matrix[:, col_idx].toarray().ravel().astype(float)
            else:
                col_vec = matrix[:, col_idx].astype(float)

            if std_y == 0 or np.std(col_vec) == 0:
                r = 0.0
            else:
                r, _ = pearsonr(col_vec, y)
                if np.isnan(r):
                    r = 0.0
            pearson_r_list.append(float(r))

        pearson_r_arr = np.array(pearson_r_list)

        # Retain terms with absolute correlation > threshold
        retained_mask = np.abs(pearson_r_arr) > threshold
        removed_mask = np.abs(pearson_r_arr) <= threshold

        retained_terms = feature_names[retained_mask].tolist()
        removed_terms = feature_names[removed_mask].tolist()

        # Build full_report DataFrame sorted descending by pearson_r
        report_df = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_r_arr
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        retained_preview = retained_terms[:20]
        removed_preview = removed_terms[:20]

        result_id = session.next_result_id("pearson_filter")
        args = {"target_class": target_class, "threshold": threshold}

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "retained_terms": len(retained_terms),
            "removed_terms": len(removed_terms),
            "retained_list": retained_preview,
            "removed_list": removed_preview,
            "retained_list_preview_size": len(retained_preview),
            "removed_list_preview_size": len(removed_preview),
            "truncated": len(retained_terms) > 20 or len(removed_terms) > 20,
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args=args,
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Calculates Spearman rank correlation between term counts and a target class indicator.

        Args:
            target_class: Category name to correlate against.
            threshold: Terms with absolute Spearman correlation less than or equal to
                this threshold are flagged for removal (default: 0.0).

        Returns:
            Dict containing result_id, target_class, threshold, total_terms,
            retained_terms, removed_terms, retained_list, and removed_list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature matrix found. Run build_dtm_tool first.")

        if session.labels is None or len(session.labels) != session.feature_matrix.shape[0]:
            raise ValueError("Session labels missing or length mismatch with feature matrix.")

        if session.categories is None or target_class not in session.categories:
            raise ValueError(f"Invalid target_class '{target_class}'. Options: {session.categories}")

        matrix = session.feature_matrix
        feature_names = np.array(session.feature_names)

        # Binary target indicator vector y (1.0 if label == target_class, else 0.0)
        y = (np.array(session.labels) == target_class).astype(float)

        # Compute Spearman correlation r per column without densifying the whole matrix
        spearman_r_list = []
        n_terms = matrix.shape[1]
        std_y = np.std(y)

        for col_idx in range(n_terms):
            if issparse(matrix):
                col_vec = matrix[:, col_idx].toarray().ravel().astype(float)
            else:
                col_vec = matrix[:, col_idx].astype(float)

            if std_y == 0 or np.std(col_vec) == 0:
                r = 0.0
            else:
                r, _ = spearmanr(col_vec, y)
                if np.isnan(r):
                    r = 0.0
            spearman_r_list.append(float(r))

        spearman_r_arr = np.array(spearman_r_list)

        # Retain terms with absolute correlation > threshold
        retained_mask = np.abs(spearman_r_arr) > threshold
        removed_mask = np.abs(spearman_r_arr) <= threshold

        retained_terms = feature_names[retained_mask].tolist()
        removed_terms = feature_names[removed_mask].tolist()

        # Build full_report DataFrame sorted descending by spearman_r
        report_df = pd.DataFrame({
            "term": feature_names,
            "spearman_r": spearman_r_arr
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        retained_preview = retained_terms[:20]
        removed_preview = removed_terms[:20]

        result_id = session.next_result_id("spearman_filter")
        args = {"target_class": target_class, "threshold": threshold}

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "retained_terms": len(retained_terms),
            "removed_terms": len(removed_terms),
            "retained_list": retained_preview,
            "removed_list": removed_preview,
            "retained_list_preview_size": len(retained_preview),
            "removed_list_preview_size": len(removed_preview),
            "truncated": len(retained_terms) > 20 or len(removed_terms) > 20,
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args=args,
            summary=summary,
            full_report=report_df
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
