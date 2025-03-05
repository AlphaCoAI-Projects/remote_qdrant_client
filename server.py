from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any, Union
from QdrantRetriever import AsyncQdrantRetriever
from pathlib import Path
import uvicorn

app = FastAPI(title="Qdrant Vector Store API", version="1.0.0")

# Pydantic models for request and response
class PDFData(BaseModel):
    page_no: int
    page_text: str
    page_table: str


class StoreRequest(BaseModel):
    company_id: str
    pdf_data: List[PDFData]


class DeleteRequest(BaseModel):
    company_id: str


class QueryRequest(BaseModel):
    company_id: str
    query: str
    distance_type: Literal["cosine", "euclidean", "manhattan"]
    top_k: int = 5
    score_threshold: float = 0.5


class ContiguousQueryRequest(BaseModel):
    company_id: str
    query: str
    top_k: int = 5
    distance_type: Literal["cosine", "euclidean", "manhattan"] = "cosine"
    score_threshold: float = 0.5
    k_before: int = 1
    k_after: int = 1


class ResponseModel(BaseModel):
    status: str
    message: str
    data: Optional[List[Dict[str, Any]]] = None


# Create a global instance of the retriever
retriever = AsyncQdrantRetriever()


@app.post("/store", response_model=ResponseModel)
async def store_endpoint(request: StoreRequest):
    try:
        # Convert Pydantic model to tuple format expected by retriever
        pdf_data_tuples = [
            (item.page_no, item.page_text, item.page_table) for item in request.pdf_data
        ]
        
        await retriever.store(pdf_data_tuples, request.company_id)
        
        return ResponseModel(
            status="success",
            message=f"Successfully stored {len(request.pdf_data)} documents for company: {request.company_id}",
            data=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store data: {str(e)}")


@app.post("/delete", response_model=ResponseModel)
async def delete_endpoint(request: DeleteRequest):
    try:
        response = await retriever.delete(request.company_id)
        
        return ResponseModel(
            status="success",
            message=f"Successfully deleted data for company: {request.company_id}",
            data=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete data: {str(e)}")


@app.post("/query", response_model=ResponseModel)
async def query_endpoint(request: QueryRequest):
    try:
        results = await retriever.query(
            query=request.query,
            company_id=request.company_id,
            distance_type=request.distance_type,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
        
        return ResponseModel(
            status="success",
            message=f"Query returned {len(results)} results",
            data=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query_contiguous", response_model=ResponseModel)
async def query_contiguous_endpoint(request: ContiguousQueryRequest):
    try:
        results = await retriever.query_partly_contiguous_pages(
            company_id=request.company_id,
            query=request.query,
            top_k=request.top_k,
            distance_type=request.distance_type,
            score_threshold=request.score_threshold,
            k_before=request.k_before,
            k_after=request.k_after,
        )
        
        return ResponseModel(
            status="success",
            message=f"Contiguous query returned {len(results)} results",
            data=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contiguous query failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    # Ensure collection exists on startup
    if not await retriever._has_collection():
        await retriever._create_collection()


if __name__ == "__main__":
    # Use this for local development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)