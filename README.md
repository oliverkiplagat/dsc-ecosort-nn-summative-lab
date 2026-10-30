<div align="center">

#  EcoSort: Intelligent Waste Management & RAG Assistant

<p align="center">
  <b>An end-to-end multimodal deep learning pipeline integrating CNN image classification, text categorization, and TF-IDF Retrieval-Augmented Generation (RAG) for municipal waste sorting guidelines.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%252B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/TensorFlow-MobileNetV2-orange.svg" alt="Deep Learning">
  <img src="https://img.shields.io/badge/NLP-TF--IDF%20%7C%20Scikit--Learn-green.svg" alt="NLP">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

</div>

---

##  Table of Contents
- [Project Overview](#-project-overview)
- [Project Architecture & Structure](#-project-architecture--structure)
- [Core Technical Implementation](#-core-technical-implementation)
  - [1. Multimodal Input Processing & Classification](#1-multimodal-input-processing--classification)
  - [2. Policy Document Chunking & TF-IDF Vectorization](#2-policy-document-chunking--tf-idf-vectorization)
  - [3. Retrieval-Augmented Generation (RAG) Engine](#3-retrieval-augmented-generation-rag-engine)
- [System Integration & Pipeline Flow](#-system-integration--pipeline-flow)
- [Performance Evaluation & Metrics](#-performance-evaluation--metrics)
- [Installation & Local Setup](#-installation--local-setup)
- [Technology Stack](#-technology-stack)
- [Author](#-author)

---

##  Project Overview

The **EcoSort Intelligent Waste Management Assistant** addresses municipal recycling compliance by bridging visual item recognition, natural language text analysis, and automated policy retrieval.

By unifying computer vision (MobileNetV2 CNN), supervised text classification, and vector-space document similarity (TF-IDF + Cosine Similarity), EcoSort identifies target waste items from either image uploads or text descriptions and immediately surface actionable, jurisdiction-specific disposal guidelines.

### Key Competencies Demonstrated:
- **Transfer Learning Vision Model:** Fine-tuning MobileNetV2 on the 9-class RealWaste image dataset.
- **Multimodal Feature Processing:** Harmonizing unstructured image tensors and text queries into unified system interfaces.
- **Text Vector Space Embeddings:** Chunking raw municipal policy documents into dense TF-IDF vector embeddings.
- **Retrieval-Augmented Generation (RAG):** Context-aware similarity lookup to fetch relevant disposal directives dynamically.
- **Unified Software Architecture:** Object-Oriented design wrapping models, vectorizers, and fallback guards into a single end-to-end pipeline.

---

## 📂 Project Architecture & Structure

```text
dsc-ecosort-nn-summative-lab/>
│
├── dsc-ecosort-nn-summative-lab.ipynb   # Main summative lab notebook (Models, RAG & evaluation)
├── realwaste_dataset/                   # RealWaste image corpus categorized into 9 waste classes
├── metro_city_policy_documents.csv     # Unstructured municipal recycling directives & guidelines
├── item_descriptions.csv                # Textual waste descriptions for NLP classifier training
├── requirements.txt                     # Pinned Python dependencies for exact reproducibility
└── README.md                            # Comprehensive project documentation