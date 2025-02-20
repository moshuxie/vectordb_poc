from vectordb_poc import VectorDB
if __name__ == '__main__':
    vector_db = VectorDB(model_path="/Users/mikexie/vectordb/bge-small-zh-v1.5")

    vector_db.create_db(
        csv_path="sentences.csv",
        col_name="text",
        source_name="sentences",
        batch_size=2560,
        db_path="vectordb",
        collection_name="20230916",
    )
