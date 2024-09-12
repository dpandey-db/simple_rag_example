
##### A simplified example of creating and deploying a RAG agent based on https://github.com/databricks/genai-cookbook

### How to Use

##### Option 1:
if you already have a vector search index:

1. update deploy_agent_config.yaml 
2. update rag_chain_config.yaml 
3. run deploy_agent notebook

##### Option 2:
if you already have chunked content in a delta table:

1. update create_vector_search_index_config.yaml.
2. run create_vector_search_index notebook (or create index through UI).
3. update deploy_agent_config.yaml 
4. update rag_chain_config.yaml 
5. run deploy_agent notebook


##### Option 3:
you are starting from scratch

1. update chunk_pdf_config.yaml
2. upload pdf docs to volume specified in chunc_pdf_config.yaml
3. run chunk_pdf
4. update create_vector_search_index_config.yaml.
5. run create_vector_search_index notebook (or create index through UI).
6. update deploy_agent_config.yaml 
7. update rag_chain_config.yaml 
8. run deploy_agent notebook



### Notes
1. the vector search index and vector search endpoint name are defined in both create_vector_search_index_config.yaml and in rag_chain_config.yaml.  Make sure they are the same.
2. the embedding model name appears in both chunk_pdf_config.yaml and create_vector_search_index_config.yaml
3. maybe consolidate all yaml files into one?
