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
- **Content**: Scientific abstracts from different fields. It consists of 63,357 arXiv papers converted to multi-markdown (.mmd) format. It includes original arXiv article IDs, titles, abstracts, authors, publication dates, URLs, and corresponding markdown files published between January 2023 and October 2023.
- **Size**: 63,357 rows

## Preprocessing
To prepare the dataset for topic modeling, several preprocessing steps are implemented:
1. **Tokenization and Lemmatization**: Abstracts were tokenized and lemmatized for consistency.
2. **Stop Words and Unwanted Tokens Removal**: Removed common stop words, numbers, and other irrelevant content.

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

After running the **BERTopic** model, **108 topics** were identified. Here’s a sample of some of the topics that were discovered:

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
  [Heatmap Visualization](https://raw.githubusercontent.com/khushidubeyokok/BERTopic/refs/heads/main/Visualizations/heatmap.html)
  ![](https://github.com/user-attachments/assets/8c21f5fe-e2fd-49a3-8a4f-2aedd3df747e)

- **`topic_model.visualize_barchart()`**
  [Barchart Visualization](https://raw.githubusercontent.com/khushidubeyokok/BERTopic/refs/heads/main/Visualizations/barchart.html)
  ![Screenshot 2024-11-09 002815](https://github.com/user-attachments/assets/15e99096-4b2c-415f-bd5f-db1535c30213)

- **`topic_model.visualize_topics()`**
[Intertopic Visualization](https://raw.githubusercontent.com/khushidubeyokok/BERTopic/refs/heads/main/Visualizations/intertopic.html)
![Screenshot 2024-11-09 002841](https://github.com/user-attachments/assets/fc8e238b-3410-4510-a602-e1340f4f6a5b)

- **`topic_model.visualize_hierarchy()`**
[Hierarchy Visualization](https://raw.githubusercontent.com/khushidubeyokok/BERTopic/refs/heads/main/Visualizations/hierarchy.html)
![Screenshot 2024-11-09 003004](https://github.com/user-attachments/assets/6d341136-3082-482e-af18-0cb31d5b3b83)

## Practical Applications: 
### 1.Monthly Topic Analysis

[Monthly Topic Analysis notebook](https://github.com/khushidubeyokok/BERTopic/blob/main/1_Monthly_trend_analysis.ipynb) demonstrates the practical application of topic modeling results to analyze monthly publication frequencies of research papers.

In this notebook, users can:
- Analyze monthly publication frequencies for specific topics by entering a topic number.
- Provide insights on how certain topics gained popularity over time, enabling researchers to identify hot topics in their field.
- Enable users to track progress in specific research areas, fostering data-driven decisions for future work.
- Suggest areas where more research might be needed or where emerging topics are gaining attention.
  
![image](https://github.com/user-attachments/assets/6e109ac8-f098-4d05-b688-f1db09aac19a)

This analysis, based on the `titles_topics_probabilities.csv` and publication dates from our original dataset, offers deeper insights into the temporal patterns of research publications.

### 2.Title vs Abstract Analysis

[Title vs Abstract Analysis](https://github.com/khushidubeyokok/BERTopic/blob/main/2_Title_vs_Abstract_Analysis.ipynb) explores the correlation between research paper titles and abstracts.

In this notebook, users can :
- Analyze the relationship between titles and their corresponding abstracts to identify patterns in the data.
- Investigate how well the topics extracted from the abstracts align with the titles of the papers.
- Use the correlation analysis to improve the coherence of topics and guide future research

  ![Screenshot 2025-01-07 174057](https://github.com/user-attachments/assets/a6cd00af-7172-4b97-9bc3-8ab0457cd5b6)
  ![Screenshot 2025-01-07 174106](https://github.com/user-attachments/assets/240bec7b-426d-4602-b554-05b8843daa78)
  ![Screenshot 2025-01-07 174120](https://github.com/user-attachments/assets/a34b711a-a0a0-4f04-9608-bc817b29b123)

  This notebook provides valuable insights into how titles and abstracts might influence each other and contributes to a deeper understanding of topic modeling.

## Conclusion
This project successfully applied BERTopic to uncover meaningful topics within a research paper dataset. By refining the topic model parameters and using targeted preprocessing, we achieved a balance between topic granularity and interpretability. The practical applications of this project are diverse:
- **Monthly Topic Analysis**: By tracking topic popularity over time, researchers can identify emerging areas of interest and monitor trends in their field. This feature enables data-driven decisions about where to focus future research efforts.
- **Correlation Analysis Between Titles and Abstracts**: This analysis helps uncover relationships between paper titles and their corresponding abstracts, improving the coherence of topics and contributing to better topic interpretations.
Overall, this tool can be valuable for researchers to quickly identify relevant areas, explore trends in scientific literature, and gain deeper insights into the connections between paper titles and abstracts. :)

## Future Work
- **Parameter Optimization**: Experiment with other sentence transformers for potentially finer-grained topic distinctions.
- **Outlier Analysis**: Develop a more robust method to handle outliers, potentially enriching topic coherence.
- **Expansion**: Scale the model to larger datasets as computational resources allow.


