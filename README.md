# Rufus - Intelligent Web Scraper

This repository implements an intelligent web scrapper that takes in a query value amd relevant sites, scraps the site and return the relevant content based on the query for downstream tasks such as RAG.

# Usage

```python
from Rufus import client

urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
queries = ["What is an agent in AI?"]

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

print(results)
```

# Approach
- The following client is built by taking large scale queries into mind. Most of the functions implemented in the `Client` can run asynchronously and the database is also built to handle large scale queries.
- The web loader uses `playwright` to load the webpage and extract the content. One of the main reasons for using `playwright` is that it can load the webpage as a user would and hence can load the dynamic content as well.
- Then after loading the webpage, we filter out the nested urls from the webpage with the help of LLMs and based on queries extract the most relevant nested urls.
- After retreiving the nested web page and the current web pages, we recusrsively split the content into smaller chunks. This has been done to make sure that the LLMs doesn't lose context in the middle of the content and can give better results.
- After spilliting, the content is passed through the LLMs to extract the most relevant content out of the corpus. This has been done in order to make the scrapper robust to dynamic changes in the webpage.
- The extracted content is then stored in the database for further use.
- Finally, we use the similarity search to find the most relevant content based on the query.