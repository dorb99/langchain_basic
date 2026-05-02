import os

import typer
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.panel import Panel

from config.settings import SETTINGS
from config.tracing import init_tracing
from llm.factory import get_embeddings

app = typer.Typer(help="Ingest docs into Chroma for assistant v2")
console = Console()


def ingest(docs_path: str) -> int:
    docs = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
        separators=["\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for ch in chunks:
        src = ch.metadata.get("source")
        if isinstance(src, str) and src:
            ch.metadata["source"] = os.path.basename(src)

    # Wipe the existing collection so re-ingesting doesn't append duplicates.
    try:
        Chroma(
            persist_directory=SETTINGS.chroma_persist_dir,
            collection_name=SETTINGS.chroma_collection,
            embedding_function=get_embeddings(),
        ).delete_collection()
    except Exception:
        pass

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=SETTINGS.chroma_persist_dir,
        collection_name=SETTINGS.chroma_collection,
    )
    return len(chunks)


@app.command()
def run(docs_path: str = typer.Option(..., "--docs-path", help="Path to docs directory")) -> None:
    init_tracing()

    count = ingest(docs_path)
    console.print(Panel(f"Stored {count} chunks", title="Ingestion v2", border_style="cyan"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
