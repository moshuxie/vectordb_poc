from vectordb_poc import VectorDB
if __name__ == '__main__':
    vector_db = VectorDB(model_path="/Users/mikexie/vectordb/bge-small-zh-v1.5")

    res = vector_db.get_similar_sentences(
        query_texts="nba",
        n_results=4,
        db_path="vectordb",
        collection_name="20230916",
    )
    print(res)