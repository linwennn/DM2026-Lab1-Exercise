from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def binarize_labels_tool(target_category: str) -> dict:
        """One-hot / one-vs-rest binarizes labels for target_category (1 if target, else 0).

        Args:
            target_category: Category label string to encode as positive class (1).

        Returns:
            A summary dictionary containing result_id, target_category, positive_count,
            negative_count, and binarized_labels preview (first 20 labels).
        """
        if session.labels is None:
            raise ValueError("No session.labels available. Load a dataset with labels first.")

        labels_list = list(session.labels)
        unique_labels = sorted(list(set(labels_list)))

        if target_category not in unique_labels:
            raise ValueError(
                f"Target category '{target_category}' not found in session.labels. "
                f"Available labels: {unique_labels}"
            )

        if session.dataframe is not None and len(labels_list) != len(session.dataframe):
            raise ValueError(
                f"Length mismatch: session.labels length ({len(labels_list)}) "
                f"does not match session.dataframe length ({len(session.dataframe)})."
            )

        # Build one-vs-rest binary label list (1 for target_category, 0 otherwise)
        binary_list = [1 if lbl == target_category else 0 for lbl in labels_list]

        positive_count = sum(binary_list)
        negative_count = len(binary_list) - positive_count

        # Store binary labels and target category separately in session.artifacts
        session.artifacts["binarized_labels"] = binary_list
        session.artifacts["binarized_target_category"] = target_category

        binary_preview = binary_list[:20]

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "target_category": target_category,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "binarized_labels": binary_preview,
            "binarized_labels_preview_size": len(binary_preview),
            "truncated": len(binary_list) > 20,
        }

        session.store_result(
            tool_name="binarize_labels_tool",
            args={"target_category": target_category},
            summary=summary,
        )

        return summary

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: int = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
    ) -> dict:
        """Reduces document-term matrix dimensionality to 2D using PCA, t-SNE, or UMAP and sets a category-colored scatter plot figure.

        Args:
            method: Dimensionality reduction technique: 'pca', 'tsne', or 'umap'. Default 'pca'.
            n_components: Target coordinate dimensions. Must be 2. Default 2.
            random_state: Seed for reproducible reduction. Default 42.
            perplexity: t-SNE perplexity parameter. Must be strictly between 0 and n_samples. Default 30.0.
            n_neighbors: UMAP n_neighbors parameter. Must be strictly between 1 and n_samples (2 <= n_neighbors < n_samples). Default 15.

        Returns:
            A summary dictionary containing result_id, method, coordinates preview (first 20 rows), and method-specific metadata.
        """
        if session.feature_matrix is None:
            raise ValueError("No session.feature_matrix available. Build a document-term matrix first.")

        if n_components != 2:
            raise ValueError(f"n_components must be exactly 2, got {n_components}.")

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        n_samples, n_features = X.shape

        if n_samples == 0:
            raise ValueError("session.feature_matrix is empty (0 documents).")

        # Category labels for scatter plot color alignment
        if session.labels is None:
            raise ValueError("No session.labels available. Category labels are required for colored plot alignment.")

        if len(session.labels) != n_samples:
            raise ValueError(
                f"Category labels length ({len(session.labels)}) does not match feature_matrix row count ({n_samples})."
            )

        cat_labels = list(session.labels)

        # Dimensionality reduction computation
        explained_variance_ratio = None

        if method == "pca":
            if min(n_samples, n_features) < n_components:
                raise ValueError(
                    f"PCA requires min(n_samples, n_features) >= {n_components}, "
                    f"got n_samples={n_samples}, n_features={n_features}."
                )
            try:
                from sklearn.decomposition import PCA
            except ImportError as err:
                raise ImportError("scikit-learn is required for PCA method.") from err

            model = PCA(n_components=n_components, random_state=random_state)
            coords_arr = model.fit_transform(X)
            explained_variance_ratio = [float(v) for v in model.explained_variance_ratio_]

        elif method == "tsne":
            if not (0 < perplexity < n_samples):
                raise ValueError(
                    f"t-SNE perplexity ({perplexity}) must satisfy 0 < perplexity < n_samples ({n_samples})."
                )
            try:
                from sklearn.manifold import TSNE
            except ImportError as err:
                raise ImportError("scikit-learn is required for t-SNE method.") from err

            model = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            coords_arr = model.fit_transform(X)

        elif method == "umap":
            if not (2 <= n_neighbors < n_samples):
                raise ValueError(
                    f"UMAP n_neighbors ({n_neighbors}) must satisfy 2 <= n_neighbors < n_samples ({n_samples})."
                )
            try:
                import umap
            except ImportError as err:
                raise ImportError("umap-learn package is required for UMAP method.") from err

            model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
            coords_arr = model.fit_transform(X)

        else:
            raise ValueError(f"Unsupported reduction method '{method}'. Choose 'pca', 'tsne', or 'umap'.")

        coords_list = coords_arr.tolist()
        session.artifacts["reduced_coordinates"] = coords_list

        # Mandatory category-colored 2D scatter plot figure creation
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_df = pd.DataFrame({
            "Dim 1": coords_arr[:, 0],
            "Dim 2": coords_arr[:, 1],
            "Category": cat_labels,
        })

        sns.scatterplot(
            data=plot_df,
            x="Dim 1",
            y="Dim 2",
            hue="Category",
            style="Category",
            s=100,
            ax=ax,
        )
        ax.set_title(f"2D Dimension Reduction ({method.upper()})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        plt.tight_layout()

        session.pending_figure = fig

        coords_preview = coords_list[:20]

        result_id = session.next_result_id(f"reduce_dimensions_{method}")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "total_documents": n_samples,
            "coordinates": coords_preview,
            "coordinates_preview_size": len(coords_preview),
            "truncated": n_samples > 20,
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
            },
            summary=summary,
        )

        return summary

    return [binarize_labels_tool, reduce_dimensions_tool]
