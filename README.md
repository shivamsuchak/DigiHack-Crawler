# Webpage Clustering and Analysis with BERT and Clustering Algorithms

This project scrapes web pages, extracts content, generates embeddings using BERT, and clusters the content using K-Means and DBSCAN. Additionally, it performs Named Entity Recognition (NER) on the summarized content and evaluates clustering performance using silhouette scores.

---

## Features

- **Web Scraping**: Extracts hyperlinks and content from the `infosys.com/industries/` page.
- **BERT Embeddings**: Converts webpage content into BERT embeddings.
- **Clustering**: Clusters the BERT embeddings using K-Means and DBSCAN.
- **Named Entity Recognition (NER)**: Extracts entities like product names, dates, and locations from the summarized content.
- **Summarization**: Summarizes webpage content using a BART model.
- **Silhouette Score**: Evaluates the performance of clustering algorithms.

---


## Usage

1. **Run the Script**:
 ```bash
 python your_script_name.py
 ```

2. **Functionality**:
- **Web Scraping**: The script fetches the main page of `infosys.com/industries/` and extracts the first 30 hyperlinks.
- **BERT Embedding**: Generates BERT embeddings for the webpage content.
- **Clustering**: Performs clustering using K-Means and DBSCAN.
- **Named Entity Recognition (NER)**: Extracts named entities from the summarized text.
- **Evaluation**: Computes and displays the silhouette scores for the clustering algorithms.

3. **Output**:
The script generates a JSON file named `webpage_data_with_clusters.json` containing the clustered data, summaries, and named entities extracted from the content.

---

## Example

Below is an example of a JSON object in the output file:

```json
{
 "cluster_id": 1,
 "url": "https://www.infosys.com/industries/example-page.html",
 "title": "Example Page",
 "headings": "Heading 1 Heading 2",
 "paragraphs": "This is a summarized version of the content.",
 "links": "https://www.example.com",
 "entities": {
     "ORG": ["Infosys"],
     "DATE": ["2024"],
     "LOCATION": ["Bangalore"]
 }
}

