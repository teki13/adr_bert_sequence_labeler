# adr_bert_sequence_labeler

## Overview

In this project, we undertook several tasks to enhance the extraction of Adverse Drug Reactions (ADRs). Initially, we examined ADRMine, which utilizes CRF and word2vec embeddings for ADR extraction. We sought to improve upon this method by employing BERT embeddings in the ADRMine framework. Moreover, we fine-tuned BERT for sequence labeling specifically to identify ADRs.

Furthermore, we implemented baseline information retrieval models such as BM25 and TF-IDF to assess the performance of the various models.

Through a comparative study of all the models, we observed that the fine-tuned BERT model achieved the highest performance. This outcome aligns with our expectations, as BERT has the ability to capture contextual information effectively.

For detailed information on the results and a comprehensive explanation of each approach, please refer to the accompanying report: "Report_Transformers-based_approach-ADR.pdf"

## Structure

The repository is structured in the following way:

- Report_Transformers-based_approach-ADR.pdf: The report encompasses comprehensive explanations of each implemented model, provides a summary of the results obtained, and delves into a thorough literature review on the topic
- code: this folder contains the supporting code for reproducing the project
- data: some of the data used as part of the project

## Code

The code folder consists of 6 different python scripts, namely:
  
- 1_creating_dataset.py: this script contains the code that cleans all 4 datasets used for this project and combined them into a single dataset. This script needs to be run before the rest
- 2_baseline.py: this script contains the implementation of the information retrieval models, specifically TF-IDF and BM25
- 3_BERT_model.py: this script includes the implementation of the fine-tuned BERT model for sequence labeling
- 4_word2vec_embeddings_and_crf.py: this script contains the replica of the ADRMine model
- 5_bert_embeddings.py: this script includes the creating of the bert embeddings. This script needs to be run before running the 6_crf_with_bert_embeddings.py one
- 6_crf_with_bert_embeddings.py: this script implements the ADRMine model with BERT embeddings instead of word2vec embeddings

## Data

This folder contains some of the data relevant for the project. It includes the ADR lexicon needed for the baseline information retrieval models, as well as the combined and clead dataset which is used for the analysis. It does not include the 4 original raw datasets which were used to create the combined and cleaned one. 

## Requirements

- Python
- Pytorch

## Collaborators

- Teona Ristova
- Marta Emili Garcia Segura
- Karina Gabor
- Giomaria Murgia
