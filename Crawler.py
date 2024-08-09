import requests
from bs4 import BeautifulSoup
import json
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel, pipeline
import torch
import numpy as np
import pandas as pd
from requests.exceptions import RequestException

#Fetch the main page
url = "https://www.infosys.com/industries/"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise error for HTTP issues
    soup = BeautifulSoup(response.text, 'html.parser')
except RequestException as e:
    print(f"Failed to fetch {url}: {e}")
    soup = None

#Extract all hyperlinks
links = []
if soup:
    try:
        for a_tag in soup.find_all('a', href=True):
            full_url = requests.compat.urljoin(url, a_tag['href'])
            links.append(full_url)

        #Limit to the first 100 hyperlinks
        links = links[:30]
    except Exception as e:
        print(f"Failed to extract links: {e}")

#Scrape each linked page, extract raw content, and store in JSON format
data = []
for link in links:
    try:
        page_response = requests.get(link, headers=headers)
        page_response.raise_for_status()  # Raise error for HTTP issues
        page_soup = BeautifulSoup(page_response.text, 'html.parser')

        # Extract raw content
        raw_content = page_soup.get_text(separator=' ', strip=True)

        # Store in dictionary
        page_data = {
            'url': link,
            'title': page_soup.title.string if page_soup.title else "No title",
            'raw_content': raw_content
        }
        data.append(page_data)
    except RequestException as e:
        print(f"Failed to fetch {link}: {e}")
        continue

#Generate BERT embeddings for each document
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    """Generate BERT embedding for a given text."""
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Generate embeddings for all documents
embeddings = []
for item in data:
    try:
        embedding = get_bert_embedding(item['raw_content'])
        embeddings.append(embedding)
    except Exception as e:
        print(f"Failed to generate embedding for {item['url']}: {e}")

# Convert embeddings to a NumPy array
X = np.array(embeddings)

#Normalize the embeddings
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Clustering with K-means
try:
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
except Exception as e:
    print(f"Failed to perform K-means clustering: {e}")
    kmeans_labels = np.zeros(X_scaled.shape[0])

# Clustering with DBSCAN
try:
    dbscan = DBSCAN(eps=0.7, min_samples=5, metric='cosine').fit(X_scaled)
    dbscan_labels = dbscan.labels_
except Exception as e:
    print(f"Failed to perform DBSCAN clustering: {e}")
    dbscan_labels = np.zeros(X_scaled.shape[0])

#Analyze and compare clusters
for i, item in enumerate(data):
    item['kmeans_cluster'] = int(kmeans_labels[i])
    item['dbscan_cluster'] = int(dbscan_labels[i])

# Initialize the summarization pipeline with BERT model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)

def chunk_text(text, max_length=512):
    """Splits the input text into chunks no longer than max_length tokens."""
    tokens = text.split()
    for i in range(0, len(tokens), max_length):
        yield " ".join(tokens[i:i + max_length])

def summarize_text(text, max_length=512):
    """Summarize the input text by chunking if necessary."""
    if len(text.split()) > max_length:
        chunks = chunk_text(text, max_length=max_length)
        summary = " ".join([summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks])
    else:
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    return summary

def extract_entities(text):
    """Extract named entities from the text using NER."""
    entities = ner(text)
    extracted_entities = {}
    for entity in entities:
        entity_type = entity['entity_group']
        if entity_type not in extracted_entities:
            extracted_entities[entity_type] = []
        extracted_entities[entity_type].append(entity['word'])
    
    # Deduplicate and format the extracted entities
    for entity_type in extracted_entities:
        extracted_entities[entity_type] = list(set(extracted_entities[entity_type]))
    
    return extracted_entities

# Function to fetch and parse additional elements from the webpages
def fetch_and_parse_elements(data):
    elements_data = []
    for item in data:
        cluster_id = item.get('kmeans_cluster') if item.get('kmeans_cluster') is not None else item.get('dbscan_cluster')
        
        if cluster_id is None:
            continue  # Skip if no cluster ID is found
        
        try:
            response = requests.get(item['url'])
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract elements
            title = soup.title.string if soup.title else "No title"
            headings = ' '.join(h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
            paragraphs = ' '.join(p.get_text(strip=True) for p in soup.find_all('p'))
            links = ' '.join(a['href'] for a in soup.find_all('a', href=True))

            # Summarize paragraph content
            summarized_paragraph = summarize_text(paragraphs)

            # Extract named entities from the summarized content
            entities = extract_entities(summarized_paragraph)

            # Append to list with cluster information
            elements_data.append({
                'cluster_id': cluster_id,
                'url': item['url'],
                'title': title,
                'headings': headings,
                'paragraphs': summarized_paragraph,
                'links': links,
                'entities': entities
            })
        except RequestException as e:
            print(f"Failed to fetch or parse {item['url']}: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    return elements_data

#Fetch elements from each URL and associate them with the corresponding cluster
elements_data = fetch_and_parse_elements(data)

# Convert the collected data into a DataFrame
df = pd.DataFrame(elements_data)
df.to_json('webpage_data_with_clusters.json', orient='records', lines=True)

# Load and pretty-print the JSON data
try:
    with open('webpage_data_with_clusters.json', 'r') as file:
        json_obj = [json.loads(line) for line in file]
    print(json.dumps(json_obj, indent=4))
except json.JSONDecodeError as json_err:
    print(f"JSON decode error: {json_err}")

#Compare the clustering performance using silhouette scores
try:
    dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
    print(f"\nSilhouette Score (DBSCAN): {dbscan_silhouette}")
    print(f"Silhouette Score (K-means): {kmeans_silhouette}")
except Exception as e:
    print(f"Failed to compute silhouette scores: {e}")
