# So In order to gather information from thje web, we need three things
# 1. Search: Search the web for given query
# 2. Load: Load the web page
# 3. Extract: Extract the information from the web page

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import re
from tqdm import tqdm

# Defining the search query
urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
queries = ["What is an agent in AI?"]


class Client:
    def __init__(
        self, urls, queries, llm="gpt-4o-mini", vectorstore_path="./chroma_db_oai"
    ):
        self.urls = urls
        self.queries = queries

        # Defining the schema of how we want to extract the information
        self.schema = {
            "properties": {
                "title": {"type": "string", "selector": "title"},
                "content": {"type": "string", "selector": "p"},
            }
        }

        self.schema_nested_urls = {
            "properties": {
                "url": {"type": "string", "description": "The URL of the webpage"}
            },
            "required": ["url"],
            "description": "Selects the relevant URLs from the webpage with maximum of `max_selected_urls` with context dependent on the query",
            "queries": queries,
            "max_selected_urls": 5,
        }

        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        self.vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(), persist_directory=vectorstore_path
        )

    def compute_nested_urls(self, corpus):
        """
        The function computes the nested urls from the webpage
        """
        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        nested_urls = url_pattern.findall(corpus)

        # cleaning
        nested_urls = [url.split(")")[0] for url in nested_urls]

        extract_llm = create_extraction_chain(
            schema=self.schema_nested_urls, llm=self.llm
        )
        extracted = extract_llm.run(Document(page_content=" ".join(nested_urls)))

        final_urls = [url["url"] for url in extracted]

        return final_urls

    def load_nested_page(self, url):
        """
        The function loads the nested webpage and returns the parsed html
        """

        loader = AsyncChromiumLoader(url)
        transformer = BeautifulSoupTransformer()
        docs_transformed = loader.load()
        return transformer.transform_documents(docs_transformed)

    def load_page(self, url):
        """
        The function loads the webpage and returns the parsed html
        """

        if type(url) != list:
            url = [url]

        loader = AsyncChromiumLoader(url)
        transformer = BeautifulSoupTransformer()
        docs_transformed = loader.load()

        corpus = docs_transformed[0].page_content
        relevant_nested_urls = self.compute_nested_urls(corpus)
        relevant_nested_pages = self.load_nested_page(relevant_nested_urls)

        return transformer.transform_documents(docs_transformed) + relevant_nested_pages

    def chunk_webpage(self, docs, chunk_size=10000, chunk_overlap=100):
        """
        The function chunks the webpage into smaller parts with overlap between different chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splits = splitter.split_documents(docs)
        return splits

    def extract_content(self, splits):
        """
        The function uses LLM to extract the content from the webpage and parse that in specific schema
        `create_extraction_chain` ensures that the dynamic content change is handled properly
        i.e. even if the page changes dynamically the extraction chain will still work
        """

        extract_contents_llm = create_extraction_chain(schema=self.schema, llm=self.llm)
        extracted = []

        for split in tqdm(splits, desc="Extracting content", total=len(splits)):
            contents = extract_contents_llm.run(split)
            extracted += contents
        return extracted

    def store_data(self, data):
        """
        The function stores the extracted data in the database
        """

        # TODO: This can be optimized by adding multiple documents at once
        for dt in data:
            try:
                docs = Document(
                    page_content=dt[0]["content"], metadata={"title": dt[0]["title"]}
                )
                self.vectorstore.add_documents(ids=[dt[0]["title"]], documents=[docs])
            except Exception as e:
                print(e)


if __name__ == "__main__":
    extracted_data = []
    client = Client(urls, queries)
    for i in range(len(urls)):
        docs = client.load_page(urls[i])
        splits = client.chunk_webpage(docs)
        extracted = client.extract_content(splits)
        client.store_data(extracted)
        extracted_data.append(extracted)

    results = []
    for query in queries:
        result = client.vectorstore.similarity_search(query, k=1)
        results.append(result)

    print(extracted_data, results)
