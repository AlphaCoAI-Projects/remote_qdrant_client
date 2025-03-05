# Qdrant Retriever

This repository implements a retriever for PDF data using Qdrant and Sentence Transformers. It enables storing and querying embeddings generated from PDF pages, allowing efficient retrieval of text data based on a query.

## Requirements

The project depends on the following Python libraries:

- `qdrant-client`: Python client for Qdrant vector search engine.
- `sentence-transformers`: A library for using pre-trained transformer models to generate sentence embeddings.

You can install the required dependencies with:

```bash
pip install -r requirements.txt
docker compose up -d
```

## Qdrant Retriever

The core class of this project is `QdrantRetriever`, which performs two main actions:

- **Store**: Takes a tuple containing PDF data and stores embeddings in a Qdrant collection.
- **Query**: Takes a query string and retrieves the most relevant documents from the stored embeddings.

### Class Overview

#### `QdrantRetriever`

This class handles interactions with Qdrant, including creating collections, storing data, and querying for relevant documents.

### Key Constants

- `COLLECTION_NAME`: The name of the collection in Qdrant where the data is stored.
- `VECTOR_SIZE`: Size of the embeddings (set to 1024 for this implementation).
- `PAGE_NO`, `PAGE_TEXT`, `PAGE_TABLE`: Constants used for parsing the structure of PDF data.

### Methods

#### `__init__(self)`

- Initializes the Qdrant client connection (`http://127.0.0.1:6333`).

#### `store(self, pdf_data: tuple[int, str, str], company_id: str) -> None`

- Stores PDF data in Qdrant after generating embeddings for each page.
- **Arguments**:
  - `pdf_data`: A tuple where each entry contains page number, page text, and table data (in that order).
  - `company_id`: The company ID used to filter the data when querying.
- Creates the collection if it doesn't exist.

#### `query(self, query: str, company_id: str, distance_type: Literal["cosine", "euclidean", "manhattan"], top_k: int = 5, score_threshold: float = 0.5)`

- Queries the Qdrant collection with an input query string and retrieves the most relevant pages.
- **Arguments**:
  - `query`: The query string to search for.
  - `company_id`: The company ID to filter the data by.
  - `distance_type`: The distance metric for retrieval. Options are `"cosine"`, `"euclidean"`, and `"manhattan"`.
  - `top_k`: The number of results to retrieve (default is 5).
  - `score_threshold`: Minimum score threshold for filtering results.
  
  Returns a list of relevant page texts.

#### Private Methods

- `_has_collection(self) -> bool`: Checks if the collection exists in Qdrant.
- `_create_collection(self) -> None`: Creates a Qdrant collection with the required vector configuration.
- `_get_embeddings(self, pdf_data: list[str]) -> list[list]`: Generates embeddings for each page of the PDF.
- `_get_vector_points(self, embeddings: list[list], pdf_data: tuple[int, str, str], company_id: str) -> models.PointsList`: Creates Qdrant vector points from the embeddings and PDF data.
- `_get_query_vector(self, query: str) -> list`: Converts the query string into a vector using the Sentence Transformer model.

## Example Usage

### 1. Initialize QdrantRetriever

You can initialize the `QdrantRetriever` class like this:

```python
from qdrant_retriever import QdrantRetriever

retriever = QdrantRetriever()
```

### 2. Store PDF Data

You can store PDF data by providing a tuple for each page with page number, text content, and table data:

```python
pdf_data = [
    (1, "Page 1 text content", "Page 1 table data in csv"),
    (2, "Page 2 text content", "Page 2 table data csv"),
]
company_id = "company123"
retriever.store(pdf_data, company_id)
```

In this example:
- `pdf_data` contains a list of tuples, where each tuple represents one page.
- `company_id` is the identifier for the company you're associating with the data. It will be used later for filtering when querying.

### 3. Query Data

To query the stored PDF data, you can use the `query` method. For example, to search for relevant text regarding a company's revenue:

```python
query = "What is the revenue of company123?"
results = retriever.query(query, company_id, distance_type="cosine", top_k=3, score_threshold=0.8)
print(results)
```

In this example:
- `query` is the text you're searching for.
- `company_id` is used to filter the data to only include relevant entries.
- `distance_type` specifies the similarity measure. It can be `"cosine"`, `"euclidean"`, or `"manhattan"`.
- `top_k` specifies the number of results you want to retrieve (in this case, 3).
- `score_threshold` specifies the minimum required similarity score for the result to be included.
- The results are a list of page text excerpts that are most relevant to your query.

### 4. Storing New Data After Creation

If you wish to store more data after creating the initial collection, simply call the `store()` method with new `pdf_data`:

```python
new_pdf_data = [
    (3, "Page 3 new text content", "Page 3 new table data in csv"),
]
new_company_id = "company456"
retriever.store(new_pdf_data, new_company_id)
```

This will append more data to the Qdrant collection.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
