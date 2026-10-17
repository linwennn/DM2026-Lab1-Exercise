import json
import math
import os
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def check_missing_tool(text_column: str = "text") -> dict:
        """Checks session.dataframe for missing, NaN, or empty text values.

        Args:
            text_column: The column name to check for missing/empty text values. Default "text".

        Returns:
            A summary dictionary containing result_id, text_column, n_missing, and missing_indices.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Load a dataset first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(
                f"Column '{text_column}' not found in session.dataframe. "
                f"Available columns: {list(session.dataframe.columns)}"
            )

        series = session.dataframe[text_column]

        # Missing condition: NaN/null OR string stripped of whitespace is empty
        is_missing = series.isna() | (series.astype(str).str.strip() == "")
        missing_indices = session.dataframe.index[is_missing].tolist()
        n_missing = len(missing_indices)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "text_column": text_column,
            "n_missing": n_missing,
            "missing_indices": missing_indices,
        }

        session.store_result(
            tool_name="check_missing_tool",
            args={"text_column": text_column},
            summary=summary,
        )

        return summary

    @tool
    def check_duplicates_tool(
        text_column: str = "text",
        drop: bool = False,
    ) -> dict:
        """Checks for duplicate documents in session.dataframe and optionally drops all copies (keep=False).

        Args:
            text_column: Column name to evaluate for duplicates. Default "text".
            drop: If True, drops ALL copies of duplicate rows (keep=False) from session.dataframe. Default False.

        Returns:
            A summary dictionary containing result_id, n_duplicates, duplicate_indices,
            n_rows_dropped, and n_rows_after.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Load a dataset first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(
                f"Column '{text_column}' not found in session.dataframe. "
                f"Available columns: {list(session.dataframe.columns)}"
            )

        # Count duplicate extras using keep='first' (matches TEST_FIXTURE.md #3 n_duplicates=1)
        extras_mask = session.dataframe.duplicated(subset=[text_column], keep="first")
        n_duplicates = int(extras_mask.sum())

        # All duplicate row copies (for keep=False dropping and complete index reporting)
        all_dups_mask = session.dataframe.duplicated(subset=[text_column], keep=False)
        duplicate_indices = session.dataframe.index[all_dups_mask].tolist()

        if drop and len(duplicate_indices) > 0:
            n_rows_before = len(session.dataframe)
            session.dataframe.drop_duplicates(subset=[text_column], keep=False, inplace=True)
            session.dataframe.reset_index(drop=True, inplace=True)
            n_rows_dropped = n_rows_before - len(session.dataframe)
            action_taken = f"Dropped {n_rows_dropped} rows (keep=False)."
        else:
            n_rows_dropped = 0
            action_taken = "Checked duplicates (no rows dropped)."

        n_rows_after = len(session.dataframe)
        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "text_column": text_column,
            "n_duplicates": n_duplicates,
            "duplicate_indices": duplicate_indices,
            "n_rows_dropped": n_rows_dropped,
            "action_taken": action_taken,
            "n_rows_after": n_rows_after,
        }

        session.store_result(
            tool_name="check_duplicates_tool",
            args={"text_column": text_column, "drop": drop},
            summary=summary,
        )

        return summary

    @tool
    def sample_data_tool(n: int, random_state: int = 42) -> dict:
        """Subsamples n rows from session.dataframe for exploration or fast prototyping.

        Args:
            n: Number of document rows to sample.
            random_state: Seed for reproducible random sampling. Default 42.

        Returns:
            A summary dictionary containing result_id, n_requested, n_sampled,
            sample_indices, and sample_data.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Load a dataset first.")

        sampled_df = session.dataframe.sample(n=n, random_state=random_state)
        sample_indices = sampled_df.index.tolist()
        sample_data = json.loads(sampled_df.to_json(orient="records"))

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "n_requested": n,
            "n_sampled": len(sample_indices),
            "sample_indices": sample_indices,
            "sample_data": sample_data,
        }

        session.store_result(
            tool_name="sample_data_tool",
            args={"n": n, "random_state": random_state},
            summary=summary,
        )

        return summary

    @tool
    def tokenize_tool(text_column: str = "text") -> dict:
        """Tokenizes text into unigrams using nltk.word_tokenize on lowercased text and stores unigrams in session.dataframe.

        Args:
            text_column: Column name containing document text to tokenize. Default "text".

        Returns:
            A summary dictionary containing result_id, text_column, n_documents,
            total_tokens, and sample_tokens.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Load a dataset first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(
                f"Column '{text_column}' not found in session.dataframe. "
                f"Available columns: {list(session.dataframe.columns)}"
            )

        def tokenize_text(val):
            if pd.isna(val) or not str(val).strip():
                return []
            return nltk.word_tokenize(str(val).lower())

        unigrams = session.dataframe[text_column].apply(tokenize_text).tolist()
        session.dataframe["unigrams"] = unigrams
        session.artifacts["unigrams"] = unigrams

        total_tokens = sum(len(doc_tokens) for doc_tokens in unigrams)
        sample_tokens = unigrams[:3]

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "text_column": text_column,
            "n_documents": len(unigrams),
            "total_tokens": total_tokens,
            "sample_tokens": sample_tokens,
        }

        session.store_result(
            tool_name="tokenize_tool",
            args={"text_column": text_column},
            summary=summary,
        )

        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> dict:
        """Lists files and directories in a repository subdirectory for dataset discovery.

        Args:
            subdirectory: Path relative to repository root to list. Default "" (repo root).

        Returns:
            A summary dictionary containing result_id, subdirectory, directories, and files.
        """
        target_path = os.path.join(os.getcwd(), subdirectory) if subdirectory else os.getcwd()

        if not os.path.exists(target_path):
            raise ValueError(f"Directory '{subdirectory}' does not exist.")

        if not os.path.isdir(target_path):
            raise ValueError(f"Path '{subdirectory}' is not a directory.")

        entries = os.listdir(target_path)
        dirs = sorted([e for e in entries if os.path.isdir(os.path.join(target_path, e))])
        files = sorted([e for e in entries if os.path.isfile(os.path.join(target_path, e))])

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": dirs,
            "files": files,
        }

        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary,
        )

        return summary

    @tool
    def describe_data_tool(
        text_column: str = "text",
        category_column: str = "category_name",
    ) -> dict:
        """Computes text length summary statistics overall and per category, and generates a boxplot.

        Args:
            text_column: Column containing document text. Default "text".
            category_column: Column containing category label names. Default "category_name".

        Returns:
            A summary dictionary containing result_id, overall_stats, and per_category_stats.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Load a dataset first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(
                f"Column '{text_column}' not found in session.dataframe. "
                f"Available columns: {list(session.dataframe.columns)}"
            )

        def clean_val(v):
            if pd.isna(v) or (isinstance(v, float) and not math.isfinite(v)):
                return None
            return float(v)

        # Treat missing / NaN values as empty string "" before computing character lengths
        text_series = session.dataframe[text_column].fillna("").astype(str)
        text_length = text_series.apply(len)

        # Temporary local DataFrame for grouping and plotting (without mutating session.dataframe)
        temp_df = pd.DataFrame({"text_length": text_length})
        if category_column in session.dataframe.columns:
            temp_df[category_column] = session.dataframe[category_column]

        # Overall summary stats
        overall_stats = {k: clean_val(v) for k, v in temp_df["text_length"].describe().to_dict().items()}

        # Per-category summary stats
        per_category_stats = {}
        if category_column in temp_df.columns:
            grouped = temp_df.groupby(category_column)["text_length"]
            for cat_name, group in grouped:
                cat_desc = group.describe().to_dict()
                per_category_stats[str(cat_name)] = {k: clean_val(v) for k, v in cat_desc.items()}

        # Build boxplot figure
        fig, ax = plt.subplots(figsize=(6, 4))
        if category_column in temp_df.columns:
            sns.boxplot(data=temp_df, x=category_column, y="text_length", ax=ax)
            ax.set_title("Document Text Length by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=temp_df, y="text_length", ax=ax)
            ax.set_title("Document Text Length Distribution")

        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats,
        }

        session.store_result(
            tool_name="describe_data_tool",
            args={"text_column": text_column, "category_column": category_column},
            summary=summary,
        )

        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> dict:
        """Peeks at the top n rows and dataset metadata of session.dataframe.

        Args:
            n: Number of leading document rows to inspect. Default 5.

        Returns:
            A summary dictionary containing result_id, n_rows, n_columns, columns,
            n_requested, and head_records.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Load a dataset first.")

        if n <= 0:
            raise ValueError(f"Parameter n must be greater than 0, got {n}.")

        df = session.dataframe
        head_df = df.head(n)
        head_records = json.loads(head_df.to_json(orient="records"))

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "n_requested": n,
            "head_records": head_records,
        }

        session.store_result(
            tool_name="inspect_data_tool",
            args={"n": n},
            summary=summary,
        )

        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        tokenize_tool,
        list_files_tool,
        describe_data_tool,
        inspect_data_tool,
    ]
