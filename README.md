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

  
