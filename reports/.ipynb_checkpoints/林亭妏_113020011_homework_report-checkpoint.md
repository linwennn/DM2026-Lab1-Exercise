# Exploring Reddit Stock Sentiment

## Dataset and text representation
I analyzed 847 Reddit posts and comments using `load_dataset_1`: 315 negative (-1.0), 423 neutral (0.0), and 109 positive (1.0). The classes were imbalanced, with positive sentiment being the smallest group. `check_missing_3` found no missing texts, while `check_duplicates_4` identified 23 duplicate occurrences beyond the first copies. I retained all rows for this baseline analysis, so repeated texts may influence frequencies and patterns. In `describe_data_5`, the overall median text length was 79 characters and the maximum was 2,753. These texts were much shorter than the earlier 20-Newsgroups dataset, whose median was 1,235 characters.

![Text length by sentiment](../plots/林亭妏_homework_plot_describe_data_5.png)

## DTM and feature filtering
`tokenize_6` produced 26,386 tokens. Separately, `build_dtm_7` created a raw-count DTM with 847 documents, 4,346 terms, and 17,193 non-zero entries, approximately 99.53% sparse. `term_frequency_8` reported 21,047 total term occurrences; its top ten terms were common function words, led by “the,” “to,” and “and.” The token and DTM totals differ because the tools use different tokenization rules. With variance > 0.01, `variance_filter_9` retained 506 candidate terms and flagged 3,840 for removal. For positive sentiment versus the other classes, an absolute correlation threshold of 0.1 retained 41 terms with Pearson (`pearson_filter_10`) and 39 with Spearman (`spearman_filter_11`). These filters reported candidates without modifying the DTM. The Pearson plot included “huge,” “nice,” and “potential,” alongside less obviously emotional terms such as “main” and “technology.” Association with the label does not establish causation or predictive accuracy.

![Top positive Pearson correlations](../plots/林亭妏_homework_plot_turn26.png)

## Frequent pattern mining
I applied category-local TF-IDF filtering and Top-K mining with k=10 to positive posts (`mine_patterns_12`). All ten returned patterns were single terms. Examples were “just” with support 15, “good” with support 12, and “doing” with support 9; “market,” “silver,” and “economy” also appeared in the results. Support counts documents containing a pattern rather than total word occurrences. Unlike the header-dominated patterns in the earlier newsgroup analysis, these results included conversational and financial vocabulary. However, frequency within positive posts alone does not show that a term is specific to positive sentiment; comparison with neutral and negative posts would be needed.

## Dimensionality reduction
I compared PCA (`reduce_dimensions_pca_13`), t-SNE with perplexity=30 and random_state=42 (`reduce_dimensions_tsne_14`), and UMAP with n_neighbors=15 and random_state=42 (`reduce_dimensions_umap_15`) on the unchanged DTM. PCA's first two components explained approximately 27.00% and 4.99% of the variance, totaling 31.99%. Most documents overlapped near the main concentration of points, with a few distant outliers. The t-SNE and UMAP plots also showed extensive mixing among the three sentiment classes rather than three clearly separated groups. These visualizations do not measure classification accuracy or prove that sentiment cannot be predicted from the original data.

![PCA projection](../plots/林亭妏_homework_plot_reduce_dimensions_pca_13.png)

![t-SNE projection](../plots/林亭妏_homework_plot_reduce_dimensions_tsne_14.png)

![UMAP projection](../plots/林亭妏_homework_plot_reduce_dimensions_umap_15.png)

## Additional checks and conclusions
`binarize_labels_16` produced 109 positive indicators and 738 non-positive indicators, preserving the original three-class labels. Here, non-positive includes both neutral and negative posts. `cosine_similarity_17` returned 0.0 for documents 0 and 1, labeled negative and neutral respectively in `inspect_data_2`. This indicates no shared non-zero terms in their DTM vectors, not necessarily opposite sentiment or unrelated meaning. Overall, the tools ran on the Reddit dataset without dataset-specific code changes. The analysis highlighted short texts, class imbalance, duplicate content, and substantial sentiment overlap in the projections. A follow-up should inspect duplicate-label consistency and evaluate candidate features on held-out data before making claims about sentiment prediction.