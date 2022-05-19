from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import hdbscan
import umap
import torch
from tqdm.notebook import tqdm
import numpy as np
from scipy import spatial


class Recommender:
    @torch.no_grad()
    def __init__(self, emb_path, txt_uncleaned, topics, embeddings_file=None, txt_cleaned=None, params=None,
                 model_path=None, additional_data=None):
        self.embedder = SentenceTransformer(emb_path, device="cpu")
        self.additional = None
        if model_path is None:
            self.params = params
            umap_model = umap.UMAP(**self.params[0], random_state=42)
            hdbscan_model = hdbscan.HDBSCAN(**self.params[1], prediction_data=True)
            self.topic_model = BERTopic(embedding_model=self.embedder, umap_model=umap_model,
                                        hdbscan_model=hdbscan_model,
                                        **self.params[2], verbose=True, calculate_probabilities=True)
        else:
            self.topic_model = BERTopic().load(model_path)
        self.uncleaned_texts = np.array(txt_uncleaned)
        if embeddings_file is None:
            self.embeddings = np.array(self.embedder.encode(txt_cleaned, show_progress_bar=True))
        else:
            self.embeddings = np.load(embeddings_file)
        if additional_data is not None:
            self.additional = np.array(additional_data)
        self.topics = np.array(topics)


    def __compute_simmilarity(self, txt_emb, q_emb):
        return 1 - spatial.distance.cosine(txt_emb, q_emb)


    @torch.no_grad()
    def recommend(self, queries, n_t=3):
        result = dict()
        queries_embs = self.embedder.encode(queries)
        for i, query_emb in tqdm(enumerate(queries_embs), total=len(queries), desc='Getting recommendations'):
            tops, sim = self.topic_model.find_topics(queries[i])
            sims = []
            to_use_ind = np.arange(0, self.topics.shape[0])[self.topics == tops[0]]
            tmp_embs = self.embeddings[to_use_ind]
            tmp_add = None
            if self.additional is not None:
                tmp_add = self.additional[to_use_ind]
            tmp_uncleaned_texts = self.uncleaned_texts[to_use_ind]
            for text_emb in tmp_embs:
                sims.append(self.__compute_simmilarity(text_emb, query_emb))
            final_ids = np.argsort(np.array(sims))[-n_t:][::-1]
            if self.additional is not None:
                result[queries[i]] = list(zip(tmp_uncleaned_texts[final_ids].tolist(), tmp_add[final_ids]))
            else:
                result[queries[i]] = tmp_uncleaned_texts[final_ids].tolist()
        return result

