from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langsmith import traceable

from config.settings import SETTINGS
from llm.factory import get_embeddings, get_llm
from prompts.templates import RAG_PROMPT
from retrieval.query_plan import plan_from_llm
from retrieval.rerank import rerank_documents


def _get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=SETTINGS.chroma_persist_dir,
        collection_name=SETTINGS.chroma_collection,
        embedding_function=get_embeddings(),
    )


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def retrieval_chain():
    vectorstore = _get_vectorstore()

    # Closure keeps `vectorstore` out of the traced inputs (it isn't JSON-serializable).
    @traceable(run_type="retriever", name="ChromaRetrieve")
    def chroma_retrieve(
        search_query: str,
        metadata_filter: dict[str, str] | None,
    ) -> list[Document]:
        kwargs: dict = {"k": SETTINGS.retriever_top_k}
        if metadata_filter:
            kwargs["filter"] = metadata_filter
        return vectorstore.similarity_search(search_query, **kwargs)

    def retrieve_context(question: str) -> str:
        plan = plan_from_llm(question)
        candidates = chroma_retrieve(plan.search_query, plan.metadata_filter)
        if not candidates:
            return ""
        top_docs = rerank_documents(plan.search_query, candidates, SETTINGS.rerank_top_n)
        return _format_docs(top_docs)

    return (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: retrieve_context(x["question"]))
        )
        | RAG_PROMPT
        | get_llm()
        | StrOutputParser()
    ).with_config(
        {"run_name": "RetrievalChain", "metadata": {"step": "retrieval"}}
    )
