from ltrpy.dataset import load as ltrpy_load
from rulpy.pipeline import task
from os.path import join
import logging


@task(use_cache=True)
async def load(dataset_file, normalize=True, filter_queries=False):
    logging.info(f"Loading dataset {dataset_file} (normalize={normalize}, filter={filter_queries})")
    return ltrpy_load(dataset_file, normalize=normalize, filter_queries=filter_queries)


@task
async def load_train(dataset_path):
    return await load(join(dataset_path, "train.txt"))


@task
async def load_vali(dataset_path):
    return await load(join(dataset_path, "vali.txt"), filter_queries=True)


@task
async def load_test(dataset_path):
    return await load(join(dataset_path, "test.txt"), filter_queries=True)

