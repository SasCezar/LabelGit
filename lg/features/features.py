import os
import re
from abc import abstractmethod, ABC

import fasttext as ft
import numpy as np
import sourcy
import spacy
from gensim.models import KeyedVectors
from more_itertools import flatten

from lgio.graph_load import ArcanGraphLoader
from utils import check_dir


class FeatureExtraction(ABC):
    def __init__(self, model="en_trf_bertbaseuncased_lg", method=None, stopwords=None):
        try:
            self.nlp = spacy.load(model, disable=["ner", "textcat", "parser"])
        except:
            pass
        self.method = method
        if not stopwords:
            stopwords = set()
        self.stopwords = stopwords

    @abstractmethod
    def get_embeddings(self, graph):
        raise NotImplemented()

    @staticmethod
    def split_camel(name):
        splitted = re.sub('([A-Z][a-z]+)|_', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
        return splitted

    @staticmethod
    def save_features(features, path, file):
        out = os.path.join(path, file)
        with open(out, "wt", encoding="utf8") as outf:
            for name, cleanded, embedding in features:
                if not isinstance(embedding, list):
                    embedding = embedding.tolist()
                rep = " ".join([str(x) for x in embedding])
                line = name + " " + rep + "\n"
                outf.write(line)

    def extract(self, project_name, graph_path, out_path, sha=None, num=None, clean_graph=False):
        graph = ArcanGraphLoader(clean=clean_graph).load(graph_path)
        features_out = os.path.join(out_path, "embeddings", self.method)
        features = self.get_embeddings(graph)
        check_dir(features_out)
        features_name = f"{project_name}-{num}-{sha}.vec" if sha and num else f"{project_name}.vec"
        self.save_features(features, features_out, features_name)


class DocumentFeatureExtraction(FeatureExtraction):
    def __init__(self, model="en_trf_bertbaseuncased_lg", method="document", preprocess=True, stopwords=None):
        super().__init__(model, method, stopwords)
        self.scp = sourcy.load("java")
        self.preprocess = preprocess

    def get_embeddings(self, graph):
        for node in graph.vs:
            path = node['filePathReal']

            if not os.path.isfile(path):
                continue

            identifiers = self.get_identifiers(path)

            text = " ".join(identifiers)
            embedding = self._create_embedding(text)

            yield path, path, embedding

    @staticmethod
    def read_file(filename):
        with open(filename, "rt", encoding="utf8") as inf:
            text = inf.read()

        return text

    def get_identifiers(self, path):
        text = self.read_file(path)

        doc = self.scp(text)

        ids = [self.split_camel(x.token) for x in doc.identifiers]
        ids = [x.lower() for x in set(flatten(ids)) if x.lower() not in self.stopwords]

        return ids

    def _create_embedding(self, text):
        return self.nlp(text).vector


class FastTextExtraction(DocumentFeatureExtraction):
    def __init__(self, model="wiki.en.bin", method="fastText", preprocess=True, stopwords=None):
        super().__init__(model, method, stopwords)
        self.nlp = ft.load_model(model)
        self.scp = sourcy.load("java")
        self.preprocess = preprocess

    def get_embeddings(self, graph):
        for node in graph.vs:
            path = node['filePathReal']

            if not os.path.isfile(path):
                continue

            identifiers = self.get_identifiers(path)

            text = " ".join(identifiers)
            embedding = self.nlp.get_sentence_vector(text)

            yield node['filePathRelative'], path, embedding


class Code2VecExtraction(DocumentFeatureExtraction):
    def __init__(self, model="code2vec.vec", method="code2vec", preprocess=True, stopwords=None):
        super().__init__(model, method, stopwords)
        self.nlp = KeyedVectors.load_word2vec_format(model)
        self.scp = sourcy.load("java")
        self.preprocess = preprocess

    def _create_embedding(self, text):
        words = text.split(" ")
        embeddings = [self.nlp.get_vector(x) for x in words]

        doc_representation = np.mean(embeddings, axis=0)

        return doc_representation

    def get_identifiers(self, path):
        text = self.read_file(path)

        doc = self.scp(text)

        ids = [x.token.lower() for x in doc.identifiers if x.token.lower() not in self.stopwords]
        ids = [x for x in ids if x and x in self.nlp.vocab]

        return ids
