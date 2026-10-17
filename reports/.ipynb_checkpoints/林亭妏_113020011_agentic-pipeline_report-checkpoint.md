# Exploring Four 20-Newsgroups Categories

## Dataset and representation
I analyzed 2,257 documents from four 20-Newsgroups categories: alt.atheism (480), comp.graphics (584), sci.med (594), and soc.religion.christian (599), loaded as `load_dataset_1` in the final session. Earlier checks found no missing or duplicate texts. Using default parameters, `build_dtm_2` created a document-term matrix with 2,257 rows and 35,788 terms, with approximately 99.55% sparsity. The representation retained common words and message headers, so the analysis reflects both document content and metadata.

## Feature filtering and correlation
I compared three filtering methods on the unchanged baseline DTM. With a variance threshold of 0.01, `variance_filter_3` retained 5,354 terms and flagged 30,434 for removal. For sci.med, using an absolute correlation threshold of 0.1, `pearson_filter_4` retained 291 terms, while `spearman_filter_5` retained 421. These tools reported candidate features rather than modifying the DTM. Spearman retained more terms, but this alone does not demonstrate better features or predictive performance. The Pearson visualization included terms such as “pitt,” “gordon,” “geb,” and “banks,” suggesting that author or signature information may contribute to category association. This interpretation would require checking the source documents for confirmation.

![Top terms from the sci.med Pearson correlation result](../plots/林亭妏_agentic-pipeline_plot_turn40.png)

## Frequent pattern mining
For sci.med, I ran category-local TF-IDF filtering followed by Top-K mining with k=10 (`mine_patterns_7`). The tool returned 10 patterns. Examples included “subject” with support 594, “lines” with support 585, and the pair “subject” and “lines” with support 585. These frequent patterns mainly describe message structure rather than medical topics. High support within sci.med does not establish category specificity, because the same headers may also occur frequently in other categories.

## Dimensionality reduction
I compared PCA (`reduce_dimensions_pca_8`), t-SNE with perplexity=30 and random_state=42 (`reduce_dimensions_tsne_9`), and UMAP with n_neighbors=15 and random_state=42 (`reduce_dimensions_umap_10`). PCA's first two components explained approximately 65.94% and 4.33% of the variance, totaling 70.27%. All three projections showed substantial category overlap, although some category-related structure was visible. PCA concentrated most documents near the origin, with a few distant outliers. In t-SNE, comp.graphics appeared more concentrated on the right, while sci.med was more common in the upper region. UMAP showed comp.graphics more concentrated toward the lower right and soc.religion.christian toward the upper left. However, neither nonlinear projection formed four clearly separated category clusters, and these visual patterns do not establish classification accuracy. These results were based on raw term counts, so document length and common vocabulary are possible influences. Separately, the AgentDev t-SNE fixture #8 coordinate mismatch remains unresolved; successful execution on this dataset does not resolve that validation issue.

![PCA projection](../plots/林亭妏_agentic-pipeline_plot_reduce_dimensions_pca_8.png)

![t-SNE projection](../plots/林亭妏_agentic-pipeline_plot_reduce_dimensions_tsne_9.png)

![UMAP projection](../plots/林亭妏_agentic-pipeline_plot_reduce_dimensions_umap_10.png)

## Additional checks and conclusions
I created a one-versus-rest sci.med indicator with `binarize_labels_11`, obtaining 594 positive and 1,663 negative labels while preserving the original labels and DTM. I also compared documents 0 and 1, both from comp.graphics, using `cosine_similarity_12`; their cosine similarity was approximately 0.3532. One pair cannot represent similarity across an entire category. Overall, the analysis showed that frequent terms, correlated features, and frequent patterns answer different questions. Metadata can strongly influence these results, so a useful follow-up would compare the baseline with a representation that removes headers and common words before judging topic-related structure.
