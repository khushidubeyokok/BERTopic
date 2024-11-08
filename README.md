# Research Paper Topic Modeling with BERTopic
[Notebook link](https://colab.research.google.com/drive/1IxH1SUiqZwDOmbwGyfHRBT26iD_31N1O?usp=sharing)

This project utilizes the **BERTopic** model to explore topics within a research paper database, focusing on the thematic structure of scientific abstracts. The goal is to assist in topic discovery and improve the accessibility of research insights.


## Project Overview
This repository provides a comprehensive walkthrough of using BERTopic for topic modeling on a research paper dataset. The project covers:

- **Dataset Selection:** Choosing a manageable dataset from Hugging Face for efficient topic modeling on Google Colab.
- **Data Processing:** Preparing the data for BERTopic by cleaning text fields and structuring inputs.
- **Topic Modeling:** Applying BERTopic to identify distinct clusters of topics in research abstracts.
- **Insights and Findings:** Visualizing and analyzing the generated topics to understand prevalent themes.


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
```

## Results and Analysis

After running the **BERTopic** model, I identified around **108 topics**. Here’s a sample of some of the topics that were uncovered:

| **Title**                              | **Topic**                  | **Probability** |
|----------------------------------------|----------------------------|-----------------|
| "Dynamics of Polymer Ejection from a Nano-Sphere "          | 61-ploymer-cell-membrane                | 0.98            |
| "Enhacning Health data interoperability with large language models : A FHIR study"       | 39-clinical-medical-health        | 1.0        |
| "Benford's Law under Zeckendorf expansion" |27-integer-number-sum-prime            | 0.93         |

### Observations:

* **Topic Diversity**: The topics cover various research areas, indicating a broad application of the model.
* **Outlier Removal**: A threshold was set to filter out low-confidence classifications to improve the overall quality of the topics.
* **Visualization**: The results were visualized using the built-in functions of **BERTopic** to illustrate the relationships between topics.


## Visualization
We generated visualizations to explore topic distributions and hierarchical structures:
- **`topic_model.visualize_heatmap()`**
![heatmap](https://github.com/user-attachments/assets/2a4a6e79-2ca7-47ca-af36-3c65da38c7a2)

- **`topic_model.visualize_barchart()`**
![barchart](https://github.com/user-attachments/assets/6f1572d4-03c9-4d44-abcb-a35db39b6180)

- **`topic_model.visualize_topics()`**
- ![intertopic](https://github.com/user-attachments/assets/2f0241db-618c-4ddb-adcd-30b05baace38)

- **`topic_model.visualize_hierarchy()`**
![hierarchy](https://github.com/user-attachments/assets/57a55792-3883-493c-b879-ed81690c5b28)


## Conclusion
This project successfully applied BERTopic to uncover meaningful topics within a research paper dataset. By refining the topic model parameters and using targeted preprocessing, we achieved a balance between topic granularity and interpretability. This tool can be valuable for researchers to quickly identify relevant areas and explore trends in scientific literature :)

## Future Work
- **Parameter Optimization**: Experiment with other sentence transformers for potentially finer-grained topic distinctions.
- **Outlier Analysis**: Develop a more robust method to handle outliers, potentially enriching topic coherence.
- **Expansion**: Scale the model to larger datasets as computational resources allow.
