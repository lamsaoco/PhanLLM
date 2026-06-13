from gitsource import GithubRepositoryDataReader
from minsearch import Index

def load_lessons_data():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    files = reader.read()

    documents = []

    for file in files:
        doc = file.parse()
        documents.append(doc)

    return documents

def build_index(documents):
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )

    index.fit(documents)
    
    return index