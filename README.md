# Beyond Leaf and Word: A Multimodal Analysis of Nahua Botanical Knowledge in the De la Cruz-Badiano Herbal

**Summary**  
This study investigates the structure of Nahua botanical knowledge as captured in the 1552 De la Cruz-Badiano Herbal, a unique document preserving Indigenous plant names and illustrations. Our goal was to determine if and how different data modalities—linguistic, morphological, and structural—align in a shared conceptual space, providing quantitative insight into this understudied worldview.

We employed a multi-modal approach, first creating a dataset with three aligned modalities: leaf shape morphometrics from hand-traced illustrations, text embeddings of corresponding English and Spanish texts, and node embeddings of a graph representing Nahuatl name co-occurrence. A three-tower CNN with a contrastive loss function was then trained to learn a common embedding space that minimized the distance between these modalities. Synthetic leaf data was used to augment the training set.

The model successfully learned a common embedding space for all three modalities. We found that the leaf shape data occupied a more distinct region of this space, while the linguistic and structural data were closely aligned. This suggests that the cultural and functional associations embedded in the Nahuatl names and texts are more strongly correlated with each other than with a plant's physical shape.

Our results indicate a sophisticated Indigenous classification system where the structure of language and name relationships provides a primary framework for understanding plants, distinct from a purely morphological basis. This research highlights the value of using computational methods to re-examine historical documents, offering a path to decolonize plant science by quantifying the richness of Indigenous botanical knowledge.

**Keywords**  
Botany, De la Cruz-Badiano Herbal, Indigenous knowledge, Libellus de Medicinalibus Indorum Herbis, Multimodal learning, Nahuatl, Plant classification

## Figures and tables  
![Alt](https://github.com/DanChitwood/DeLaCruzBadiano1552/blob/main/analysis/outputs/figures/fig_coincidence_graph.png)  

![Alt](https://github.com/DanChitwood/DeLaCruzBadiano1552/blob/main/analysis/outputs/figures/fig_combined_clustering_and_wordclouds_fasttext_umap_tficf.png)  

![Alt](https://github.com/DanChitwood/DeLaCruzBadiano1552/blob/main/analysis/outputs/figures/leaf_traces_1.png)

![Alt](https://github.com/DanChitwood/DeLaCruzBadiano1552/blob/main/analysis/outputs/figures/leaf_traces_2.png)

![Alt](https://github.com/DanChitwood/DeLaCruzBadiano1552/blob/main/analysis/outputs/figures/fig_ECT.png)  

| Metric              |   Xihuitl |   Xochitl |   Quahuitl |   Patli |   Quilitl |   micro avg |   macro avg |   weighted avg |
|:--------------------|----------:|----------:|-----------:|--------:|----------:|------------:|------------:|---------------:|
| Precision (Leaves)  |      0.99 |      0.99 |       0.99 |    0.99 |      0.99 |        0.99 |        0.99 |           0.99 |
| Recall (Leaves)     |      0.72 |      0.72 |       0.72 |    0.72 |      0.72 |        0.72 |        0.72 |           0.72 |
| F1 (Leaves)         |      0.83 |      0.83 |       0.83 |    0.83 |      0.83 |        0.83 |        0.83 |           0.83 |
| Precision (English) |      0.68 |      0.68 |       0.68 |    0.68 |      0.68 |        0.68 |        0.68 |           0.68 |
| Recall (English)    |      0.61 |      0.61 |       0.61 |    0.61 |      0.61 |        0.61 |        0.61 |           0.61 |
| F1 (English)        |      0.62 |      0.62 |       0.62 |    0.62 |      0.62 |        0.62 |        0.62 |           0.62 |
| Precision (Spanish) |      0.68 |      0.68 |       0.68 |    0.68 |      0.68 |        0.68 |        0.68 |           0.68 |
| Recall (Spanish)    |      0.56 |      0.56 |       0.56 |    0.56 |      0.56 |        0.56 |        0.56 |           0.56 |
| F1 (Spanish)        |      0.56 |      0.56 |       0.56 |    0.56 |      0.56 |        0.56 |        0.56 |           0.56 |

![Alt](https://github.com/DanChitwood/DeLaCruzBadiano1552/blob/main/analysis/outputs/figures/fig_CNN_morpheme.png)  

| Metric                      |   Image-to-Text |   Text-to-Image |   Image-to-Graph |   Graph-to-Image |   Text-to-Graph |   Graph-to-Text |
|:----------------------------|----------------:|----------------:|-----------------:|-----------------:|----------------:|----------------:|
| MAP avg.                    |          0.0512 |          0.0639 |           0.0504 |           0.0587 |          0.2529 |          0.2587 |
| MAP std.                    |          0.0051 |          0.0058 |           0.0047 |           0.0047 |          0.0047 |          0.0018 |
| Recall@1 score avg.         |          0.0161 |          0.0223 |           0.0139 |           0.0198 |          0.1025 |          0.1079 |
| Recall@1 score std.         |          0.0024 |          0.0026 |           0.0014 |           0.0046 |          0.0034 |          0.002  |
| Recall@1 above chance avg.  |         13      |         18      |          11.2    |          16      |         82.8    |         87.2    |
| Recall@5 score avg.         |          0.0685 |          0.0866 |           0.0641 |           0.0792 |          0.4093 |          0.4113 |
| Recall@5 score std.         |          0.0077 |          0.0086 |           0.009  |           0.0081 |          0.0075 |          0.0033 |
| Recall@5 above chance avg.  |         11.08   |         14      |          10.36   |          12.8    |         66.16   |         66.48   |
| Recall@10 score avg.        |          0.1247 |          0.1386 |           0.1096 |           0.1302 |          0.614  |          0.6088 |
| Recall@10 score std.        |          0.0114 |          0.0129 |           0.0085 |           0.008  |          0.009  |          0.0032 |
| Recall@10 above chance avg. |         10.08   |         11.2    |           8.86   |          10.52   |         49.62   |         49.2    |  

![Alt](https://github.com/DanChitwood/DeLaCruzBadiano1552/blob/main/analysis/outputs/figures/fig_three_towers.png)  

## Code and data  
All scripts to reproduce these analyses are found in `./analysis/scripts/`. Data to reproduce these analyses are found in `./analysis/data/`. The folders `./analysis/data/leaf_traces/`, `./analysis/data/texts/`, and `./analysis/data/texts_ES/` (leaf trace data and English and Spanish texts, respectively) will need to be unzipped before using. The FastText English and Spanish models `cc.en.300.bin` and `cc.es.300.bin` will need to be downloaded from this [link](https://fasttext.cc/docs/en/crawl-vectors.html) and placed into `./analysis/data/`. The pages of the document from which leaf traces are derived and on which they are plotted are derived from the public version of [The De la Cruz-Badiano Aztec Herbal of 1552](https://archive.org/details/aztec-herbal-of-1552) and the exact images found on [figshare](https://doi.org/10.6084/m9.figshare.29825189.v1) should be downloaded, the .zip file decompressed, and saved as the folder `pics` in `./analysis/data/`.

## Methods
The objective of this study was to analyze and classify Nahuatl plant names by integrating three distinct data modalities: linguistic, morphological, and structural. The data was derived from the 1939 William Gates English translation of the de la Cruz-Badiano Nahuatl Herbal. The pipeline consisted of unimodal analyses to characterize each data source, followed by a multi-modal approach to create a unified representation.

### Data Acquisition and Preprocessing
The primary data sources included digitized illustrations, corresponding English text descriptions, and a structural representation of plant name co-occurrence. For the morphological data, over 4,000 leaf outlines were hand-traced from illustrations using Fiji/ImageJ. These tracings were aligned using Generalized Procrustes Analysis (GPA) to normalize for position, scale, and rotation. The aligned shapes were then represented as a two-channel input for a convolutional neural network (CNN), with one channel being a radial Euler Characteristic Transform (ECT) in polar coordinates and the other a corresponding shape mask.

For the linguistic data, the English text from each plant's subchapter was aggregated. An automated Spanish translation was also generated for parallel analysis. FastText was employed to create word embeddings for both the English and Spanish texts. These embeddings were used for subsequent classification and clustering analyses.

A graph-based structural modality was created by constructing a network of Nahuatl plant names. Vertices in this graph represented unique Nahuatl names, and edges were weighted by the number of times two names co-occurred within the same subchapter. Node embeddings for this graph were then generated using the Node2Vec algorithm, with an embedding size of 128, a walk length of 30, and 200 walks per node.

### Data Augmentation
To address class imbalance and increase sample size for model training, a synthetic data generation method was used. After a Principal Component Analysis (PCA) was performed on the hand-traced leaf shapes, a Synthetic Minority Over-sampling Technique-like (SMOTE-like) approach was applied. Synthetic leaf shapes were created by sampling neighbors within the same class in the PCA space and linearly interpolating new shapes. These synthetic leaves were used to augment the training data for the image-based CNNs.

### Unimodal Analysis and Classification
Separate CNNs were trained to classify each modality into one of five main Nahuatl categories: the three main botanical categories (Xochitl, Quahuitl, Xihuitl) and two economic categories (Patli, Quilitl).

The image-based CNN used a two-channel input of radial ECT and shape mask to classify leaves. It was trained using stratified 5-fold cross-validation and employed ensemble prediction on the logits. To interpret the model's decisions, Grad-CAM visualizations were generated to highlight the image regions most relevant to classification.

A similar CNN architecture was adapted to classify the text data. This model took the FastText embeddings as input and was trained to classify the corresponding Nahuatl plant names. The key features distinguishing the classes were visualized using word clouds, which highlighted the most frequent and unique terms associated with each category.

### Multimodal Analysis and Integration
A three-tower CNN was developed to create a unified embedding space where the representations of all three modalities for the same plant were aligned. The model consisted of three parallel towers: an image tower (a 2D CNN), a text tower (a 1D CNN), and a graph tower (a linear layer). Each tower projected its modality's data into a shared 128-dimensional embedding space. The training objective was to minimize the distance between the embeddings of corresponding samples from all three modalities.

The model was trained using a combination of real and synthetic data within a stratified 5-fold cross-validation framework. A custom MultiModal Contrastive Loss function was used to simultaneously bring together the embeddings of the same plant across different modalities while pushing apart those of different plants.

The performance of this retrieval-focused model was evaluated using metrics such as Recall@k and Mean Average Precision (mAP) for all six possible retrieval directions (e.g., image-to-text, graph-to-image). To visualize the learned common embedding space, Uniform Manifold Approximation and Projection (UMAP) was applied to reduce the embeddings to two dimensions. A final figure was generated to illustrate the clustering of the different modalities in this space, and a histogram quantified the pairwise Euclidean distances between them.
