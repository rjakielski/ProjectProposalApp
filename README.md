# Project Proposal App

An interactive Streamlit application that classifies proposed oversight projects into **Top Management Priority Challenges (TMPCs)** and surfaces **similar historical OIG reports** using semantic search.

This tool helps auditors quickly understand where a proposal fits within oversight priorities and identify precedent work across oversight organizations.

---

## Features

**TMPC Classification**

* Predicts the most relevant TMPC category from:

  * project title
  * objective statement
* Uses a trained **SetFit sentence-transformer classifier**
* Displays prediction confidence scores

**Semantic Similarity Search**

* Retrieves the **top 10 most similar OIG reports**
* Uses latent semantic indexing over scraped oversight reports
* Shows:

  * similarity score
  * metadata
  * summary
  * direct report link

**Auditor-Friendly UI**

* Simple Streamlit interface
* Single-project workflow
* Immediate interpretability of results

---

## Example Workflow

1. Enter a project title
2. Enter an objective statement
3. Click **Classify**
4. Review:

   * predicted TMPC
   * model confidence
   * similar historical reports for reference

---

## Project Structure

```
Project Proposal App/
│
├── app.py
├── models/
├── data/
├── utils/
└── README.md
```

The app integrates with the companion repository:

```
Oversight Semantic Search
```

which provides the reusable semantic index and retrieval engine.

---

## Requirements

Python 3.10+ recommended

Install dependencies:

```
pip install -r requirements.txt
```

If using the semantic-search companion package locally:

```
pip install -e ../oversight-semantic-search
```

---

## Running the App

From the project root:

```
python -m streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## Semantic Search Integration

This app connects to the shared semantic-search package:

```
oversight_semantic_search
```

The search index is built from an SQLite database of scraped OIG reports.

To rebuild the index manually:

```
oversight-search build --rebuild
```

Example CLI query:

```
oversight-search query "contract oversight for other transactions"
```

Example project similarity search:

```
oversight-search project --title "Example title" --objective "Example objective"
```

---

## Model Details

The classifier uses:

* SetFit architecture
* sentence-transformer embeddings
* contrastive fine-tuning on TMPC-labeled proposals

Inputs:

```
title + objective statement
```

Outputs:

```
predicted TMPC
confidence score
```

---

## Use Cases

This application supports:

* proposal triage
* portfolio planning
* avoiding duplicate oversight work
* identifying precedent audits
* aligning projects with strategic priorities

---

## Future Improvements

Planned enhancements include:

* batch proposal classification 📊
* explanation highlights within objectives ✏️
* filtering semantic matches by agency 🏛️
* export to CSV / Excel 📄
* dashboard analytics for proposal trends 📈

---

## Author

Created by **Rachel Jakielski**

Designed to support oversight analytics, audit planning, and TMPC alignment workflows.
