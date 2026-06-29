"""
/ingest — handles file uploads, parsing, chunking, and BM25 indexing.
"""
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status

from app.core.config import get_settings
from app.core.dependencies import get_bm25_index, get_document_parser, get_chunker
from app.core.exceptions import DocumentParseError, UnsupportedFileTypeError, FileTooLargeError
from app.core.logger import get_logger
from app.models.schemas import UploadResponse, DocumentListResponse, DeleteResponse, FileStatus
from app.services.bm25_index import BM25IndexManager
from app.services.document_parser import DocumentParser
from app.services.chunker import TextChunker

router = APIRouter(prefix="/ingest", tags=["Ingest"])
logger = get_logger(__name__)


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    index: BM25IndexManager = Depends(get_bm25_index),
    parser: DocumentParser = Depends(get_document_parser),
    chunker: TextChunker = Depends(get_chunker),
):
    """
    Upload a document (PDF, TXT, MD, DOCX).
    Parses → chunks → indexes into BM25. Returns doc_id for later querying.
    """
    cfg = get_settings()

    # ── Validate extension ────────────────────────────────────────────────────
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in cfg.allowed_ext_list:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '.{ext}'. Allowed: {cfg.allowed_ext_list}",
        )

    # ── Read & validate size ──────────────────────────────────────────────────
    contents = await file.read()
    if len(contents) > cfg.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {cfg.max_file_size_mb}MB",
        )

    # ── Save to disk ──────────────────────────────────────────────────────────
    doc_id = str(uuid.uuid4())
    save_path = cfg.upload_path / f"{doc_id}_{filename}"
    save_path.write_bytes(contents)
    logger.info(f"Saved upload: {save_path.name} ({len(contents)} bytes)")

    # ── Parse + chunk + index ─────────────────────────────────────────────────
    try:
        parsed = parser.parse(save_path)
        chunks = chunker.chunk_document(parsed, doc_id)

        if not chunks:
            save_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Document parsed but no text chunks were produced.",
            )

        meta = index.add_document(
            doc_id=doc_id,
            filename=filename,
            file_type=ext,
            file_size_bytes=len(contents),
            chunks=chunks,
        )

    except (DocumentParseError, UnsupportedFileTypeError) as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.error(f"Indexing failed for {filename}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        num_chunks=len(chunks),
        status=FileStatus.INDEXED,
        message=f"Successfully indexed {len(chunks)} chunks.",
    )


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(index: BM25IndexManager = Depends(get_bm25_index)):
    """List all indexed documents."""
    docs = index.list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str, index: BM25IndexManager = Depends(get_bm25_index)):
    """Remove a document from the index."""
    try:
        index.delete_document(doc_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return DeleteResponse(doc_id=doc_id, message="Document removed from index.")
