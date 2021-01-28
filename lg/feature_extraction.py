import sys

from joblib import Parallel, delayed
from more_itertools import chunked

from features.extract import extract, rename
from features.features import *


def extract_features(in_path, repo_path, out_path, feature, workers):
    """
    Runs the extraction of features in parallel
    :param in_path:
    :param repo_path:
    :param out_path:
    :param feature:
    :param workers:
    :return:
    """
    projects = sorted([project for project in os.listdir(in_path)
                       if os.path.isdir(os.path.join(in_path, project))])

    chunks = list(chunked(projects, len(projects) // workers))

    Parallel(n_jobs=workers)(delayed(extract)(repo_path, feature, in_path, out_path, False, chunks[i])
                             for i in range(workers))


def rename_files(in_path, out_path):
    """
    Renames arcan file from graph-id-sha.grapml to project-id-sha.graphml
    :param in_path:
    :param out_path:
    :return:
    """
    projects = sorted([project for project in os.listdir(in_path)
                       if os.path.isdir(os.path.join(in_path, project))])
    rename(in_path, os.path.join(out_path, "graphs"), projects)


if __name__ == '__main__':
    arcan_path = sys.argv[1]  # Folders with arcan output
    repo_path = sys.argv[2]  # Folders with the repositories
    out_path = sys.argv[3]  # Output folder
    feature = sys.argv[4]  # Feature to extract
    workers = int(sys.argv[5])  # Number of workers
    extract_features(arcan_path, repo_path, out_path, feature, workers)
    rename_files(arcan_path, out_path)
