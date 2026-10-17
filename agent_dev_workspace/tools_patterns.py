from typing import Literal, Optional
import os
import tempfile
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> dict:
        """Mines frequent term patterns for a specific category using PAMI (FPGrowth, TopK, or MaxFPGrowth).

        Args:
            category_name: Category label string to filter documents by.
            filtering_method: Method to filter terms before mining ('variance', 'tfidf', or 'term_frequency'). Default 'tfidf'.
            algorithm: PAMI mining algorithm to run ('fpgrowth', 'topk', or 'maxfpgrowth'). Default 'fpgrowth'.
            min_sup: Minimum support threshold (required positive int for 'fpgrowth' and 'maxfpgrowth').
            k: Top-K pattern count (required positive int for 'topk').

        Returns:
            A summary dictionary containing result_id, category_name, filtering_method, algorithm,
            mined patterns preview (first 20 patterns), and metadata.
        """
        if session.dataframe is None:
            raise ValueError("No session.dataframe available. Load a dataset first.")

        if "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe missing required 'text' column.")

        if "category_name" not in session.dataframe.columns:
            raise ValueError("session.dataframe missing required 'category_name' column.")

        # Check algorithm specific parameter requirements
        if algorithm in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None or min_sup <= 0:
                raise ValueError(f"min_sup must be a positive integer when algorithm='{algorithm}', got {min_sup}.")
        elif algorithm == "topk":
            if k is None or k <= 0:
                raise ValueError(f"k must be a positive integer when algorithm='topk', got {k}.")
        else:
            raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose 'fpgrowth', 'topk', or 'maxfpgrowth'.")

        full_args = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k,
        }
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"

        def handle_empty_result(note_msg: str) -> dict:
            session.artifacts[artifact_key] = []
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "total_patterns_mined": 0,
                "patterns": [],
                "patterns_preview_size": 0,
                "truncated": False,
                "note": note_msg,
            }
            session.store_result("mine_patterns_tool", full_args, summary)
            return summary

        # 1. Filter session.dataframe to category_name
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if cat_df.empty:
            raise ValueError(f"No documents found for category_name '{category_name}'.")

        texts = cat_df["text"].dropna().tolist()
        if not texts:
            return handle_empty_result("no non-empty text found for category")

        # Build fresh category-local CountVectorizer(stop_words="english")
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm_sparse = vectorizer.fit_transform(texts)
        except ValueError:
            # e.g., empty vocabulary / only stopwords present
            return handle_empty_result("empty vocabulary after stopword removal")

        feature_names = np.array(vectorizer.get_feature_names_out())
        X = dtm_sparse.toarray()
        n_samples, n_features = X.shape

        if n_features == 0:
            return handle_empty_result("no terms survived vectorization")

        # 2. Apply chosen filtering_method
        survived_mask = np.zeros(n_features, dtype=bool)

        if filtering_method in ["variance", "term_frequency"]:
            if filtering_method == "variance":
                scores = np.var(X, axis=0)
            else:
                scores = np.sum(X, axis=0)
            
            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)
            # strictly between 5th and 95th percentile (middle 90%)
            survived_mask = (scores > p5) & (scores < p95)

        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_mat = transformer.fit_transform(dtm_sparse).toarray()
            mean_tfidf = np.mean(tfidf_mat, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            # at or above 20th percentile (top 80%)
            survived_mask = mean_tfidf >= p20

        else:
            raise ValueError(f"Unsupported filtering_method '{filtering_method}'. Choose 'variance', 'tfidf', or 'term_frequency'.")

        survived_terms = feature_names[survived_mask]
        survived_X = X[:, survived_mask]

        # 3. Check if zero terms survived filtering
        if survived_terms.size == 0:
            return handle_empty_result("no terms survived filtering")

        # 4. Convert surviving terms into a transactional DB dataframe and file
        # Transaction format: items tab-separated per document row
        transactions = []
        for row in survived_X:
            items = [survived_terms[idx] for idx, count in enumerate(row) if count >= 1]
            if items:
                transactions.append("\t".join(items))

        if not transactions:
            return handle_empty_result("no non-empty transactions produced")

        tmp_db = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv", encoding="utf-8")
        try:
            tmp_db.write("\n".join(transactions) + "\n")
            tmp_db.close()

            # Mine with PAMI algorithm (using exact class references inside modules)
            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                AlgorithmClass = getattr(FPGrowth, "FPGrowth", FPGrowth)
                model = AlgorithmClass(iFile=tmp_db.name, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE
                AlgorithmClass = getattr(FAE, "FAE", FAE)
                model = AlgorithmClass(iFile=tmp_db.name, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                AlgorithmClass = getattr(MaxFPGrowth, "MaxFPGrowth", MaxFPGrowth)
                model = AlgorithmClass(iFile=tmp_db.name, minSup=min_sup, sep="\t")

            model.startMine()
            patterns_df = model.getPatternsAsDataFrame()

        finally:
            if os.path.exists(tmp_db.name):
                os.remove(tmp_db.name)

        # 5. Extract mined patterns
        patterns_list = []
        if patterns_df is not None and not patterns_df.empty:
            # PAMI DataFrames have columns like 'Patterns' (or 'itemSets') and 'Support' (or 'support')
            pat_col = [c for c in patterns_df.columns if "pat" in c.lower() or "item" in c.lower()][0]
            sup_col = [c for c in patterns_df.columns if "sup" in c.lower()][0]

            for _, row in patterns_df.iterrows():
                raw_pat = str(row[pat_col]).strip()
                items = raw_pat.split("\t") if "\t" in raw_pat else raw_pat.split()
                support = int(row[sup_col])
                patterns_list.append({"pattern": items, "support": support})

        session.artifacts[artifact_key] = patterns_list

        patterns_preview = patterns_list[:20]
        total_patterns = len(patterns_list)

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "total_patterns_mined": total_patterns,
            "patterns": patterns_preview,
            "patterns_preview_size": len(patterns_preview),
            "truncated": total_patterns > 20,
        }

        session.store_result(
            tool_name="mine_patterns_tool",
            args=full_args,
            summary=summary,
        )

        return summary

    return [mine_patterns_tool]
