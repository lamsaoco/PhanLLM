"""Starter code for the monitoring homework.

Sets up the text-search RAG from homework 1 and a shared OpenAI client.
"""
import os
import sqlite3

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from gitsource import GithubRepositoryDataReader
from minsearch import Index

from rag_helper import RAGBase

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter, SpanExportResult

import sqlite3
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

class SQLiteSpanExporter(SpanExporter):

    def __init__(self, db_path="traces.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self):
        return True

provider = TracerProvider()
# provider.add_span_processor(
#     SimpleSpanProcessor(ConsoleSpanExporter())
# )
provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter("traces.db"))
)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")

COMMIT = "8c1834d"

# --- Load the course lessons (same as HW1, HW2, HW4) ---
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id=COMMIT,
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)

class RAGTraced(RAGBase):
    
    def search(self, query, num_results=5):
        # Wrap the search execution in a span
        with tracer.start_as_current_span("search") as span:
            # Record attributes for observability
            span.set_attribute("search.query", query)
            span.set_attribute("search.num_results", num_results)
            
            # Call the parent class search method
            return super().search(query, num_results=num_results)

    def llm(self, prompt):
        # Wrap the LLM generation in a span
        with tracer.start_as_current_span("llm") as span:
            # Record the prompt length or model name
            span.set_attribute("prompt_length", len(prompt))
            span.set_attribute("model", self.model)

            response = super().llm(prompt)

            span.set_attribute("input_tokens", response.usage.prompt_tokens)
            span.set_attribute("output_tokens", response.usage.completion_tokens)

            # Call the parent class llm method
            return response

    def rag(self, query):
        # Wrap the entire RAG pipeline in a root span
        with tracer.start_as_current_span("rag") as span:
            # Record the initial user query
            span.set_attribute("rag.query", query)
            
            # Call the parent class rag method
            # This will internally call our traced search and llm methods
            return super().rag(query)

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
rag = RAGTraced(index=index, llm_client=client)

if __name__ == "__main__":
    query = "How does the agentic loop keep calling the model until it stops?"
    answer = rag.rag(query)
    print(answer)
