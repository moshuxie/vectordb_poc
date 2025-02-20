import chromadb
from text2vec import SentenceModel
from typing import List
import pandas as pd

class VectorDB:
    def __init__(self, model_path: str):
        """
        :param model_path: 文本转向量模型的路径，可以是本地文件夹，也可以是 huggingface 模型名
        """
        self.model = SentenceModel(model_path)

    def get_embeddings(
        self, sentences: List[str], batch_size: int
    ) -> List[List[float]]:
        """
        :param sentences: 需要转换为向量的句子列表
        :param batch_size: 文本转换为向量的批次，即每次转换的数量
        :return: 嵌套的句子向量
        """
        # 将文本转换为向量
        embeddings = []
        
        num_batches = (len(sentences) + batch_size - 1) // batch_size  # 计算总共需要多少批次
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(sentences))
            sentence_batch = sentences[start_idx:end_idx]  # 获取当前批次的文本
            batch_embeddings = self.model.encode(sentence_batch)  # 转换当前批次的文本为向量
            embeddings.extend(batch_embeddings)  # 将转换结果添加到列表中
        embeddings = [embedding.tolist() for embedding in embeddings]
        return embeddings
    def create_db(
        self,
        csv_path: str,
        col_name: str,
        source_name: str,
        batch_size: int,
        db_path: str,
        collection_name: str,
    ):
        """
        :param csv_path: 单列数据的 csv 文件
        :param col_name: csv 文件中的列名
        :param source_name: 当前 csv 文件数据的来源
        :param batch_size: 文本转换为向量的批次，即每次转换的数量
        :param db_path: 向量数据库的路径
        :param collection_name: 向量数据库下的集合名
        :return: 向量数据库，它保存到设定的路径下
        """
        df = pd.read_csv(csv_path)
        sentences = df[col_name].tolist()
        vectors = self.get_embeddings(sentences, batch_size=batch_size)
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection(name=collection_name)
        num_batches = (len(sentences) + batch_size - 1) // batch_size
        ids = [f"id{i}" for i in range(len(sentences))]
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(sentences))
            batch_vectors = vectors[start_idx:end_idx]
            batch_sentences = sentences[start_idx:end_idx]
            metadatas = [{"source": source_name} for _ in range(len(batch_sentences))]
            batch_ids = ids[start_idx:end_idx]
            collection.add(
                embeddings=batch_vectors,
                documents=batch_sentences,
                metadatas=metadatas,
                ids=batch_ids,
            )

    def get_similar_sentences(
        self, query_texts: str, n_results: int, db_path: str, collection_name: str
    ):
        """
        :param query_texts: 单条待查询的句子
        :param n_results: 需要查询的相似句子数量
        :param db_path: 向量数据库的路径
        :param collection_name: 向量数据库下的集合名
        :return: 查询的相似句子结果
        """
        query_texts = [query_texts]
        query_embeddings = self.model.encode(query_texts)
        query_embeddings = [embedding.tolist() for embedding in query_embeddings]
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection(name=collection_name)
        results = collection.query(
            query_embeddings=query_embeddings, n_results=n_results
        )
        return results 
