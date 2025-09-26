#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from llama_index.core import Document, Settings
from typing import Any, List, Optional, Dict, OrderedDict
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
import faiss
import torch
import os
import numpy as np
import time


class TwoStageRetrieverCachedLLB():
    def __init__(self,
        faiss_index: callable,
        corpus_list: List,
        redir_table: Dict,
        cost_table: Dict,
        index_dir: str,
        embed_model: Optional[BaseEmbedding] = None,
        with_cache: Optional[bool] = False,
        cache_obj: Optional[OrderedDict] = OrderedDict(),
        cache_max_size: Optional[int] = -1,  # Total float count allowed
        prefetch: Optional[bool] = False,
        max_probe_cost: Optional[int] = -1,
        min_cache_cost: Optional[int] = 10000,
    ) -> None:
        self.faiss_index = faiss_index
        self.corpus_list = corpus_list
        self.embed_model = embed_model
        self.index_dir = index_dir
        self.redir_table = redir_table
        self.cost_table = cost_table
        self.total_generate = 0
        self.total_load = 0
        self.with_cache = with_cache
        self.cache_obj = cache_obj  # i -> embeddings (Tensor)
        self.total_cache_hit = 0
        self.total_cache_miss = 0
        self.total_non_cache = 0
        self.cache_max_size = cache_max_size
        self.max_probe_cost = max_probe_cost
        self.min_cache_cost = min_cache_cost
        self.cache_credit = {}      # i -> credit
        self.cache_item_sizes = {}  # i -> numel
        self.cache_current_size = 0

        # Initialize credit and clean undersized items
        if cache_max_size > 0:
            # Remove undersized items
            undersized_keys = []
            for k in list(self.cache_obj.keys()):
                if self.cost_table[k] < self.min_cache_cost:
                    undersized_keys.append(k)
            for i in undersized_keys:
                print("Deleting cache entry of cost: " + str(self.cost_table[i]))
                del self.cache_obj[i]
            # Recompute sizes
            for k, v in self.cache_obj.items():
                size = v.numel()
                self.cache_item_sizes[k] = size
                self.cache_current_size += size
                self.cache_credit[k] = self.cost_table.get(k, 0)  # Initialize credit

            while self.cache_current_size > self.cache_max_size:
                self._evict_landlord(None)  # Evict blindly to reduce to max size

        print("Pruned Cache Size: " + str(len(self.cache_obj)))
        print("Cache usage (numel): " + str(self.cache_current_size))

    def get_cache_dict(self) -> Dict:
        return self.cache_obj

    def _evict_landlord(self, new_key):
        """Perform landlord eviction for a new item or to trim size."""
        # Combine keys with new_key if provided
        G = {k for k in self.cache_obj.keys()}
        if new_key is not None:
            G.add(new_key)
            new_cost = self.cost_table[new_key]
            new_size = self.cache_item_sizes[new_key]
        else:
            new_cost, new_size = 0, 0

        credits = dict(self.cache_credit)
        if new_key is not None:
            credits[new_key] = new_cost * 1.0

        sizes = dict(self.cache_item_sizes)
        if new_key is not None:
            sizes[new_key] = new_size

        while sum(sizes[g] for g in G) > self.cache_max_size:
            delta = min(credits[g] / sizes[g] for g in G)
            for g in G:
                credits[g] -= delta * sizes[g]
            G = {g for g in G if credits[g] > 1e-6}

        # Remove everything not in G
        for k in list(self.cache_obj.keys()):
            if k not in G:
                self.cache_current_size -= self.cache_item_sizes.get(k, 0)
                del self.cache_obj[k]
                self.cache_credit.pop(k, None)
                self.cache_item_sizes.pop(k, None)

        return G, credits

    def _landlord_insert(self, key, tensor):
        """Insert a tensor into cache via landlord eviction logic."""
        if self.cache_max_size <= 0:
            return  # Caching disabled

        cost_f = self.cost_table[key]
        size_f = tensor.numel()
        print("Inserting embedding size: " + str(size_f))

        if size_f <= self.cache_max_size:
            self.cache_item_sizes[key] = size_f

            G, credits = self._evict_landlord(key)

            if key in G:
                self.cache_obj[key] = tensor
                self.cache_credit[key] = credits[key]
                self.cache_current_size += size_f


    def retrieve(
        self,
        query: str,
        top_k: int,
        nprobe: int,
    ) -> List[NodeWithScore]:
        if self.max_probe_cost <= 0:
            self.max_probe_cost = 30000
        print("TOTAL_PROBE_COST: " + str(self.max_probe_cost))

        embed_model = self.embed_model or Settings.embed_model
        # Throw error if there is no embedding model.
        if embed_model is None:
            raise ValueError("Embedding model is not set.")

        start = time.time()
        query_embedding = embed_model.get_text_embedding(query)
        stop = time.time()
        print("Embed gen time: " + str(stop - start))
        start = stop

        query_embedding = torch.tensor([query_embedding])
        Dis, Idx = self.faiss_index.search(query_embedding, nprobe)
        stop = time.time()
        print("1st stage search: " + str(stop - start))
        start = stop

        # Get doc index from redir table
        doc_ids = []
        embeddings_list = []
        documents = []
        total_gen_cost = 0
        for i in Idx[0]:
            doc_ids = self.redir_table[i]
            text_group = []
            for doc_id in doc_ids:
                record = self.corpus_list[doc_id]
                doc = TextNode(
                    text=record["text"], metadata={"title": record["title"], "doc_id": record["_id"]}
                )
                documents.append(doc)
                text_group.append(record["text"])

            if self.cost_table[i] > int(self.max_probe_cost / nprobe):
                print("Loading")
                # Retrieve Second-level index
                second_level_index_path = os.path.join(self.index_dir, "second_stage_" + str(i))
                # Load the index
                second_level_faiss_index = faiss.read_index(second_level_index_path)
                num_docs = second_level_faiss_index.ntotal
                dim = second_level_faiss_index.d
                new_embeddings = faiss.rev_swig_ptr(second_level_faiss_index.get_xb(), num_docs * dim).reshape(num_docs, dim)
                new_embeddings = torch.tensor(new_embeddings)
                embeddings_list.append(new_embeddings)
                self.total_load += 1
            else:
                # Look up cache
                cached_embeddings = self.cache_obj.get(i)
                if cached_embeddings is not None:
                    print("Cache hit!")
                    self.total_cache_hit += 1
                    embeddings_list.append(cached_embeddings)
                else:
                    if self.cost_table[i] > self.min_cache_cost:
                        print("Miss, generating")
                        self.total_cache_miss += 1
                    else:
                        print("NC, generating")
                        self.total_non_cache += 1

                    total_gen_cost += self.cost_table[i]
                    new_embeddings = torch.tensor(self.embed_model.get_text_embedding_batch(text_group))
                    embeddings_list.append(new_embeddings)
                    self.total_generate += 1

                    # Handle cache
                    if self.cost_table[i] > self.min_cache_cost:
                        self._landlord_insert(i, new_embeddings)

        stop = time.time()
        print("Second stage time: " + str(stop - start))
        print("Total Gen Cost: " + str(total_gen_cost))

        faiss_index_second = faiss.IndexFlatL2(len(query_embedding[0]))
        for embeddings in embeddings_list:
            faiss_index_second.add(embeddings)
        # Look up with query embedding
        Dis, Idx = faiss_index_second.search(query_embedding, top_k)
        # Construct nodes with scores
        nodes_with_scores = []
        # Loop over returned indexes
        # Add and score context to response node
        if len(Idx[0]) != 0:
            for i, D in zip(Idx[0], Dis[0]):
                nodes_with_scores.append(NodeWithScore(node=documents[i], score=D))
        return nodes_with_scores