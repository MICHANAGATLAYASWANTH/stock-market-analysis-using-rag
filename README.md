# Stock-Market-RAG — Smart Stock Market Analysis & Risk Evaluation System

**Team IDs:** SE25MAID035, SE25MAID014, SE25MAID016, SE25MAID021  

🌐 **Live Interactive Dashboard:** [https://michanagatlayaswanth.github.io/stock-market-analysis-using-rag/](https://michanagatlayaswanth.github.io/stock-market-analysis-using-rag/)  
📊 **Presentation Deck:** [StockRAG Presentation Carousel (PDF)](StockRAG_Presentation_Carousel.pdf)  
🎬 **Demo Video & Preview:** [stockrag_live_demo.mp4](stockrag_live_demo.mp4) | [stockrag_live_demo.gif](stockrag_live_demo.gif)  

A Financial Retrieval-Augmented Generation (RAG) system that retrieves semantically relevant stocks from structured financial datasets using natural-language investment queries.


Example query:

> "safe long-term banking stocks with stable returns"

The system returns financially relevant stocks along with contextual information such as:

- Sector
- Volatility
- PE Ratio
- Risk Level
- Company fundamentals
- Expected performance

---

# Project Overview

Traditional stock screeners rely mainly on filters and keyword matching. However, financial queries are often semantic and descriptive.

For example:

> "low-volatility dividend-paying technology companies with long-term growth potential"

A normal keyword-based search cannot understand the intent behind such queries.

This project solves the problem using:

- Semantic embeddings
- Financial text retrieval
- Sentence Transformers / FinBERT
- Cosine similarity search
- Financial RAG pipeline

---

# Objective

The goal of this project is to:

1. Convert natural-language investment queries into semantic embeddings.
2. Retrieve financially relevant stocks from structured market datasets.
3. Provide grounded investment insights using stock fundamentals.

---

# Features

✅ Natural-language stock search  
✅ Semantic financial retrieval  
✅ Risk-aware stock recommendations  
✅ Sector-based retrieval  
✅ Embedding-based similarity search  
✅ Financial metadata enrichment  
✅ Ablation study for evaluation  

---

# System Architecture

```text
User Query (Natural Language)
            │
            ▼
Financial Text Encoder
(SentenceTransformer / FinBERT)
            │
            ▼
Semantic Embedding Vector
            │
            ▼
Cosine Similarity Search
            │
            ▼
Top-K Relevant Stocks
(Company, Sector, PE Ratio, Volatility, Risk)
            │
            ▼
[Future Scope]
LLM-generated grounded investment recommendation
```

---

# Project Structure

```bash
.
├── stock_data_pipeline.py
├── stock_retrieval.py
├── requirements.txt
├── data/
│   ├── stock_data.csv
│   ├── stock_market_dataset.json
│   └── stock_index.pkl
└── outputs.txt
```

### File Description

| File | Description |
|------|-------------|
| `stock_data_pipeline.py` | Loads and preprocesses stock metadata |
| `stock_retrieval.py` | Builds embeddings, retrieval index, and query system |
| `requirements.txt` | Python dependencies |
| `stock_market_dataset.json` | Processed stock metadata |
| `stock_index.pkl` | Serialized semantic vector index |
| `outputs.txt` | Sample outputs and ablation results |

---

# Installation

Clone the repository:

```bash
git clone <your-github-repository-link>
cd Stock-Market-RAG
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

```txt
pandas>=2.0.0
numpy>=1.24.0
sentence-transformers>=2.2.2
transformers>=4.36.0
torch>=2.0.0
scikit-learn>=1.3.0
yfinance>=0.2.28
tqdm>=4.66.0
matplotlib>=3.7.0
nltk>=3.8.1
python-dotenv>=1.0.0
```

---

# Dataset

The project uses a structured stock market dataset containing fields such as:

- Company Name
- Sector
- Open Price
- Close Price
- High
- Low
- Volume
- PE Ratio
- News Sentiment
- Expected Performance
- Risk Level

---

# Usage

## 1️ Preprocess the Dataset

Generate cleaned stock metadata JSON:

```bash
python stock_data_pipeline.py \
--input data/stock_data.csv \
--output data/stock_market_dataset.json
```

### Arguments

| Argument | Default | Description |
|----------|----------|-------------|
| `--input` | `data/stock_data.csv` | Input stock dataset |
| `--output` | `data/stock_market_dataset.json` | Output metadata JSON |
| `--seed` | `42` | Random seed |

### Sample Output

```text
Total records : 5000
Unique sectors: 12

Risk Levels:
Low Risk      : 2100
Moderate Risk : 1900
High Risk     : 1000

Top sectors:
Technology : 1100
Banking    : 920
Energy     : 700
```

---

## 2️ Build the Semantic Financial Index

Create semantic embeddings and save the retrieval index:

```bash
python stock_retrieval.py \
--mode index \
--data data/stock_market_dataset.json
```

### Output

```text
data/stock_index.pkl
```

---

## 3️ Query Stocks Using Natural Language

Retrieve semantically relevant stocks:

```bash
python stock_retrieval.py \
--mode query \
--query "safe long-term technology stocks"
```

### Arguments

| Argument | Default | Description |
|----------|----------|-------------|
| `--query` | `"safe long-term growth stocks"` | Financial query |
| `--topk` | `5` | Number of retrieved stocks |
| `--index` | `data/stock_index.pkl` | Saved stock index |

### Sample Output

```text
Rank   Score    Company      Sector       Close      Risk
--------------------------------------------------------------
1      0.9123   Infosys      Technology   1540.20    Low Risk
       └─ Technology company with positive returns,
          stable growth, and moderate PE ratio.
```

---

# Ablation Study

Run comparison experiments between baseline ML and Financial RAG:

```bash
python stock_retrieval.py --mode ablation
```

### Results

| Query Type | Baseline ML | Financial RAG |
|------------|-------------|----------------|
| Price Prediction Accuracy | High | Moderate |
| Semantic Investment Relevance | Low | High |
| Risk Awareness | Medium | High |
| User-Friendly Insights | Low | High |

---

# Evaluation Metrics

| Metric | Milestone 1 | Final Milestone |
|--------|--------------|----------------|
| Retrieval Relevance@5 | 0.70+ | 0.90+ |
| Risk Classification Accuracy | 0.75+ | 0.90+ |
| Forecast RMSE | Baseline | Improved |
| LLM Explanation Quality | — | ROUGE-L > 0.35 |

---

# Future Scope

- Integrate Large Language Models (LLMs)
- Real-time stock market updates
- Personalized investment recommendations
- News-aware stock retrieval
- Portfolio risk analysis
- Interactive dashboard for investors

---

# Technologies Used

- Python
- Sentence Transformers
- FinBERT
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Yahoo Finance API
- Financial RAG

---

# References

1. Devlin, J. et al. (2019) — *BERT: Pre-training of Deep Bidirectional Transformers*
2. Araci, D. (2019) — *FinBERT: Financial Sentiment Analysis*
3. Lewis, P. et al. (2020) — *Retrieval-Augmented Generation*
4. Yahoo Finance API Documentation
5. NSE/BSE Historical Market Data

---

# Team Members

- SE25MAID035
- SE25MAID014
- SE25MAID016
- SE25MAID021

---

# License

This project is developed for academic and research purposes.
