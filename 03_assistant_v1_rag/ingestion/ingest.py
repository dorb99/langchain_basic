from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import typer
from rich.console import Console
from rich.panel import Panel

from config.settings import SETTINGS
from retrieval.retriever import get_embedding_function, get_vectorstore
from dotenv import load_dotenv

load_dotenv()
app = typer.Typer(help="Ingest docs into Chroma")
console = Console()


def ingest(docs_path: str) -> int:
    loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    # Always start clean: delete same-name collection if it exists.
    try:
        get_vectorstore().delete_collection()
    except Exception:
        pass

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_function(),
        persist_directory=SETTINGS.chroma_persist_dir,
        collection_name=SETTINGS.chroma_collection,
    )
    return len(chunks)

@app.command()
def run(
    docs_path: str = typer.Option(..., "--docs-path", help="Path to docs directory"),
) -> None:
    count = ingest(docs_path)
    console.print(Panel(f"Stored {count} chunks", title="Ingestion", border_style="cyan"))


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(3, "--k", min=1, help="Number of matching chunks to return"),
) -> None:
    docs = get_vectorstore().similarity_search(query, k=k)
    if not docs:
        console.print(Panel("No matching chunks found.", title="Search", border_style="yellow"))
        return

    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content.strip()
        console.print(
            Panel(
                f"[bold]Source:[/bold] {source}\n\n{content}",
                title=f"Match {idx}",
                border_style="green",
            )
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
