# simple_rag_example
simplified example of creating and deploying a RAG agent based on https://github.com/databricks/genai-cookbook

### How to Use

##### Option 1:
if you already have a vector search index:
```
update deploy_agent_config.yaml 
update rag_chain_config.yaml 
run deploy_agent notebook
```

##### Option 2:
if you already have chunked content in a delta table:
```
update create_vector_search_index_config.yaml and run create_vector_search_index notebook.  [or create index through UI]
update deploy_agent_config.yaml 
update rag_chain_config.yaml 
run deploy_agent notebook
```

##### Option 3:
you are starting from scratch
```
update chunk_pdf_config.yaml
upload pdf docs to volume specified in chunc_pdf_config.yaml
run chunk_pdf
update create_vector_search_index_config.yaml and run create_vector_search_index notebook [or create index through UI]
update deploy_agent_config.yaml 
update rag_chain_config.yaml 
run deploy_agent notebook
```


### Notes
- the vector search index and vector search endpoint name are defined in create_vector_search_index_config.yaml and in rag_chain_config.yaml.  Make sure they are the same.
- the embedding model name appears in both chunk_pdf_config.yaml and create_vector_search_index_config.yaml
- maybe consolidate all yaml files into one?
