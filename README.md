# Research Paper Topic Modeling with BERTopic

This project utilizes the **BERTopic** model to explore topics within a research paper database, focusing on the thematic structure of scientific abstracts. The goal is to assist in topic discovery and improve the accessibility of research insights.

## Project Overview
In this project, I applied **BERTopic**, a topic modeling technique that incorporates BERT embeddings, UMAP, and HDBSCAN clustering, to a dataset of research paper abstracts. The intention was to identify meaningful topics and visualize the results to facilitate better understanding of the data.

## Dataset
I selected the [neuralwork/arxiver](https://huggingface.co/datasets/neuralwork/arxiver) dataset from Hugging Face, which contains a variety of research paper abstracts. This dataset was chosen for its manageable size and relevance to my computational resources.

### Dataset Details:
- **Source**: Hugging Face Hub
- **Content**: Scientific abstracts from different fields . It consists of 63,357 arXiv papers converted to multi-markdown (.mmd) format. It includes original arXiv article IDs, titles, abstracts, authors, publication dates, URLs and corresponding markdown files published between January 2023 and October 2023.
- **Size**: 63,357 rows

## Preprocessing
To prepare the dataset for topic modeling, several preprocessing steps are implemented:
1. **Tokenization and Lemmatization**: Abstracts were tokenized and lemmatized for consistency.
2. **Stop Words and Unwanted Tokens Removal**: Removed common stop words, numbers, and other irrelevant content.
3. **Noise Reduction**: Further cleaning was performed to enhance clustering accuracy.

## Model Configuration
The **BERTopic** model was set up with the following parameters:

- **Embedding Model**: `all-MiniLM-L6-v2` for efficient sentence embeddings.
- **UMAP**: `n_neighbors=10`, `min_dist=0.1` to optimize the embedding space for clustering.
- **HDBSCAN**: `min_cluster_size=60`, `min_samples=15` to help refine the clusters formed.

### Key Parameters:
```python
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    umap_model=UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.1,
        metric='cosine'
    ),
    hdbscan_model=HDBSCAN(
        min_cluster_size=60,
        min_samples=15,
        metric='euclidean',
        cluster_selection_method='eom'
    ),
    top_n_words=10
)

