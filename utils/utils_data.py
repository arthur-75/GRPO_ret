from beir import util
from beir.datasets.data_loader import GenericDataLoader
import os
import subprocess
import json
from tqdm import tqdm


def get_data(data_set, data_path="data"):

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        data_set
    )
    data_path = util.download_and_unzip(url, data_path)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    # queries_ids = list(queries.keys())
    # queries= list(queries.values())
    # documents = [[f"{doc['title']} ,{doc['text']}"] for doc in corpus.values()]
    # document_ids= list(corpus.keys())

    return corpus, queries, qrels

def creat_index(index_path,corpus):
    if not os.path.isdir(index_path):
      os.makedirs(index_path)
    else :return None
    pyserini_jsonl = "pyserini.jsonl"
    # Build the command
    command = [
    "python", "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", index_path,
    "--index", index_path,
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "1",
    "--storePositions", "--storeDocvectors", "--storeRaw"   
    ]   

    with open(os.path.join(index_path, pyserini_jsonl), 'w', encoding="utf-8") as fOut:
        for doc_id in  tqdm(corpus):
            title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
            data = {"id": doc_id , "contents": title+" "+text}
            json.dump(data, fOut)
            fOut.write('\n')
    subprocess.run(command)
    return  None