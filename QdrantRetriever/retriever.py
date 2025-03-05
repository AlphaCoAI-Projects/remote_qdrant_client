import uuid
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from typing import Literal
import yaml
from pathlib import Path
import torch

script_directory = Path(__file__).parent
config_path = script_directory / "../config.yaml"

with open(file=config_path, mode="r") as file:
    config = yaml.safe_load(file)
    
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

class QdrantRetriever:
    COLLECTION_NAME = config["qdrant_collection_name"]
    VECTOR_SIZE = config["vector_embedding_size"]
    PAGE_NO = config["page_no"]
    PAGE_TEXT = config["page_text"]
    PAGE_TABLE = config["page_table"]
    embedding_model = SentenceTransformer(config["embedding_model"], device=device)

    def __init__(self):
        self.client = QdrantClient(location=":memory:")

    def delete(self, company_id: str):
        response = self.client.delete(
            collection_name=QdrantRetriever.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="company_id",
                            match=models.MatchValue(value=company_id),
                        ),
                    ],
                )
            ),
        )
        return response

    def store(self, pdf_data: tuple[int, str, str], company_id: str) -> None:
        if not self._has_collection():
            self._create_collection()
        pdf_data = pdf_data if isinstance(pdf_data, list) else [pdf_data]
        embeddings = self._get_embeddings(pdf_data)
        vector_points = self._get_vector_points(
            embeddings=embeddings, pdf_data=pdf_data, company_id=company_id
        )

        self.client.upsert(
            collection_name=QdrantRetriever.COLLECTION_NAME, points=vector_points
        )

    def query(
        self,
        query: str,
        company_id: str,
        distance_type: Literal["cosine", "euclidean", "manhattan"],
        top_k: int = 5,
        score_threshold: float = 0.5,
    ):
        query_vector = self._get_query_vector(query)

        retrieval_result = self.client.query_points(
            collection_name=QdrantRetriever.COLLECTION_NAME,
            query=query_vector,
            using=distance_type,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="company_id",
                        match=models.MatchText(text=company_id),
                    )
                ]
            ),
            limit=top_k,
            score_threshold=score_threshold,
            # with_payload=["page_text"]
        )
        return [hit.payload for hit in retrieval_result.points]

    def query_partly_contiguous_pages(
        self,
        company_id: str,
        query: str,
        top_k: int,
        distance_type: Literal["cosine", "euclidean", "manhattan"],
        score_threshold: float,
        k_before: int,
        k_after: int,
    ) -> list:
        # First, get the top_k seed points based on the query
        seed_points = self.query(
            query=query,
            company_id=company_id,
            distance_type=distance_type,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        # Extract the page numbers from the seed points
        seed_pages = [payload["page_no"] for payload in seed_points]

        # Generate all contiguous page numbers around each seed page
        contiguous_pages = set()
        for page in seed_pages:
            start_page = max(0, page - k_before)  # Ensure page numbers are non-negative
            end_page = page + k_after
            for p in range(start_page, end_page + 1):
                contiguous_pages.add(p)

        if not contiguous_pages:
            return []

        # Create a filter for company_id and page_no in the contiguous_pages set
        page_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="company_id", match=models.MatchValue(value=company_id)
                ),
                models.FieldCondition(
                    key="page_no", match=models.MatchAny(any=list(contiguous_pages))
                ),
            ]
        )

        # Scroll through all points that match the filter
        all_points = []
        next_offset = None
        while True:
            scroll_result, next_offset = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=page_filter,
                limit=100,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(scroll_result)
            if next_offset is None:
                break

        # Extract the payloads and avoid duplicates (using page_no as unique identifier)
        unique_pages = set()
        contiguous_payloads = []
        for point in all_points:
            page_no = point.payload["page_no"]
            if page_no not in unique_pages:
                unique_pages.add(page_no)
                contiguous_payloads.append(point.payload)

        return contiguous_payloads

    def _has_collection(self) -> bool:
        collection_names = []

        for collection in self.client.get_collections().collections:
            collection_names.append(collection.name)
        return QdrantRetriever.COLLECTION_NAME in collection_names

    def _create_collection(self) -> None:
        self.client.create_collection(
            collection_name=QdrantRetriever.COLLECTION_NAME,
            vectors_config={
                "cosine": models.VectorParams(
                    size=QdrantRetriever.VECTOR_SIZE, distance=models.Distance.COSINE
                ),
                "euclidean": models.VectorParams(
                    size=QdrantRetriever.VECTOR_SIZE, distance=models.Distance.EUCLID
                ),
                "manhattan": models.VectorParams(
                    size=QdrantRetriever.VECTOR_SIZE, distance=models.Distance.MANHATTAN
                ),
            },
        )

    def _get_embeddings(self, pdf_data: list[str]) -> list[list]:
        pdf_text = [pdf_page[QdrantRetriever.PAGE_TEXT] for pdf_page in pdf_data]
        embeddings = []

        for page_text in pdf_text:
            embedding = QdrantRetriever.embedding_model.encode(
                page_text, show_progress_bar=True
            )
            embeddings.append(embedding.tolist())
        return embeddings

    def _get_vector_points(
        self, embeddings: list[list], pdf_data: tuple[int, str, str], company_id: str
    ) -> models.PointsList:
        points = []

        for current_page, embedding in enumerate(embeddings):
            point = models.PointStruct(
                id=uuid.uuid4().hex,  # TODO: create uuid from company, year and pdf type
                vector={
                    "cosine": embedding,
                    "euclidean": embedding,
                    "manhattan": embedding,
                },
                payload={
                    "company_id": company_id,
                    "page_no": pdf_data[current_page][QdrantRetriever.PAGE_NO],
                    "page_text": pdf_data[current_page][QdrantRetriever.PAGE_TEXT],
                    "table": pdf_data[current_page][QdrantRetriever.PAGE_TABLE],
                },
            )
            points.append(point)
        return points

    def _get_query_vector(self, query: str) -> list:
        query_tensor_encoding = QdrantRetriever.embedding_model.encode(query)
        query_vector = query_tensor_encoding.tolist()
        return query_vector
