import uuid
import asyncio
from qdrant_client import AsyncQdrantClient, models
from sentence_transformers import SentenceTransformer
from typing import Literal
import yaml
from pathlib import Path
import torch

script_directory = Path(__file__).parent
config_path = script_directory / "./config.yaml"

with open(file=config_path, mode="r") as file:
    config = yaml.safe_load(file)
    
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

class AsyncQdrantRetriever:
    COLLECTION_NAME = config["qdrant_collection_name"]
    VECTOR_SIZE = config["vector_embedding_size"]
    PAGE_NO = config["page_no"]
    PAGE_TEXT = config["page_text"]
    PAGE_TABLE = config["page_table"]
    embedding_model = SentenceTransformer(config["embedding_model"], device=device)

    def __init__(self):
        self.client = AsyncQdrantClient(location=":memory:")

    async def delete(self, company_id: str):
        response = await self.client.delete(
            collection_name=self.COLLECTION_NAME,
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

    async def store(self, pdf_data: tuple[int, str, str], company_id: str) -> None:
        if not await self._has_collection():
            await self._create_collection()
            
        pdf_data = pdf_data if isinstance(pdf_data, list) else [pdf_data]
        embeddings = await self._get_embeddings(pdf_data)
        vector_points = self._get_vector_points(
            embeddings=embeddings, pdf_data=pdf_data, company_id=company_id
        )

        await self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=vector_points
        )

    async def query(
        self,
        query: str,
        company_id: str,
        distance_type: Literal["cosine", "euclidean", "manhattan"],
        top_k: int = 5,
        score_threshold: float = 0.5,
    ):
        query_vector = await self._get_query_vector(query)
        
        retrieval_result = await self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=(distance_type, query_vector),
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
        )
        return [hit.payload for hit in retrieval_result]

    async def query_partly_contiguous_pages(
        self,
        company_id: str,
        query: str,
        top_k: int,
        distance_type: Literal["cosine", "euclidean", "manhattan"],
        score_threshold: float,
        k_before: int,
        k_after: int,
    ) -> list:
        seed_points = await self.query(
            query=query,
            company_id=company_id,
            distance_type=distance_type,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        seed_pages = [payload["page_no"] for payload in seed_points]
        contiguous_pages = set()
        
        for page in seed_pages:
            start_page = max(0, page - k_before)
            end_page = page + k_after
            contiguous_pages.update(range(start_page, end_page + 1))

        if not contiguous_pages:
            return []

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

        all_points = []
        next_offset = None
        
        while True:
            scroll_result, next_offset = await self.client.scroll(
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

        unique_pages = set()
        contiguous_payloads = []
        
        for point in all_points:
            page_no = point.payload["page_no"]
            if page_no not in unique_pages:
                unique_pages.add(page_no)
                contiguous_payloads.append(point.payload)

        return contiguous_payloads

    async def _has_collection(self) -> bool:
        collections = await self.client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        return self.COLLECTION_NAME in collection_names

    async def _create_collection(self) -> None:
        await self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config={
                "cosine": models.VectorParams(
                    size=self.VECTOR_SIZE, distance=models.Distance.COSINE
                ),
                "euclidean": models.VectorParams(
                    size=self.VECTOR_SIZE, distance=models.Distance.EUCLID
                ),
                "manhattan": models.VectorParams(
                    size=self.VECTOR_SIZE, distance=models.Distance.MANHATTAN
                ),
            },
        )

    async def _get_embeddings(self, pdf_data: list) -> list[list]:
        pdf_texts = [pdf_page[self.PAGE_TEXT] for pdf_page in pdf_data]
        embeddings = await asyncio.to_thread(
            self.embedding_model.encode,
            pdf_texts,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def _get_vector_points(
        self, embeddings: list[list], pdf_data: tuple[int, str, str], company_id: str
    ) -> list[models.PointStruct]:
        points = []
        
        for idx, embedding in enumerate(embeddings):
            point = models.PointStruct(
                id=uuid.uuid4().hex,
                vector={
                    "cosine": embedding,
                    "euclidean": embedding,
                    "manhattan": embedding,
                },
                payload={
                    "company_id": company_id,
                    "page_no": pdf_data[idx][self.PAGE_NO],
                    "page_text": pdf_data[idx][self.PAGE_TEXT],
                    "table": pdf_data[idx][self.PAGE_TABLE],
                },
            )
            points.append(point)
        return points

    async def _get_query_vector(self, query: str) -> list:
        query_vector = await asyncio.to_thread(
            self.embedding_model.encode,
            query
        )
        return query_vector.tolist()

