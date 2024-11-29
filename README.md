# Qdrant - LangChain

## Installation
Clone this repository by:

HTTPS:
```
git clone https://github.com/mile888/qdrant_db.git
```
or SSH (be sure to get SSH key):
```
git clone git@github.com:mile888/qdrant_db.git
```

Create virtual anviroment and activate it.
Install poetry depedencies:
```
poetry install 
```

## Set-up docker
Set up qdrant docker
```
docker pull qdrant/qdrant
```

Run qdrant docker container
```
docker run -p 6333:6333 -v $(pwd)/qdrant_storage_volume:/qdrant/storage:z qdrant/qdrant
```

## Example
Use vector_db/qdrant_db.py scrpit for working.

Use vector_db/test.ipynb file as example how to prepare dataset, create qdrant collection and perform a search.
