import numba
import numpy as np
import json
import logging
import os
import hashlib
import copy
from time import time
from os import path
from rulpy.pipeline.task_executor import task, TaskExecutor
from argparse import ArgumentParser
from joblib import Memory
from sigir2019.evaluate import evaluate, evaluate2
from sigir2019.dataset import load_train, load_test, load_vali
from sigir2019.train import train_model as optimize_model, gradients
from sigir2019.model import load_svmrank_model
from sigir2019.util import eval_dataset, rng_seed, NumpyEncoder
from sigir2019.simulate import run_click_simulation, behaviors


def main():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                        format='[%(asctime)s] %(levelname)s %(threadName)s: %(message)s')

    # Command line parser
    cli_parser = ArgumentParser()
    cli_parser.add_argument("config", type=str)
    cli_parser.add_argument("-o", "--output", type=str)
    cli_parser.add_argument("-j", "--jobs", type=int, default=2)
    cli_parser.add_argument("-d", "--data", type=str, required=True)
    cli_parser.add_argument("-r", "--repeats", type=int, default=5)
    cli_parser.add_argument("-s", "--seed", type=int, default=4200)
    cli_parser.add_argument("-l", "--learnrates", type=str, required=True)
    args = cli_parser.parse_args()

    # Config parser with sane defaults
    parser = ArgumentParser()
    parser.add_argument("baseline", type=str)
    parser.add_argument("-z", "--zero_based", action="store_true")
    parser.add_argument("-k", "--cutoff", type=int, default=0)
    parser.add_argument("-g", "--gradient", type=str, choices=gradients.keys(),
                        default="hinge")
    parser.add_argument("-b", "--behavior", type=str, choices=behaviors.keys(),
                        default="position")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-d", "--deployments", nargs='+', type=int, default=[])
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--eps_1", type=float, default=0.1)
    parser.add_argument("--eps_2", type=float, default=1.0)

    # Load configurations to run
    configurations = []
    with open(args.config, 'rt') as f:
        for line in f:
            configurations.append(parser.parse_args(line.strip().split(" ")))

    with TaskExecutor(max_workers=args.jobs, memory=Memory("cache", compress=6)) as e:
        results = [
            run_experiment(data=args.data, conf=configuration, learnrates=args.learnrates, repeats=args.repeats, seed_base=args.seed)
            for configuration in configurations
        ]
    results = [r.result for r in results]
    
    with open(args.output, "wt") as f:
        json.dump(results, f, cls=NumpyEncoder)


@task
async def run_experiment(data, conf, learnrates, repeats, seed_base=4200):

    # Get learn rates lookup string
    with open(learnrates, 'rt') as f:
        learnrates = json.load(f)
    true_eta = 0.0 if conf.behavior == "perfect" else conf.eta
    lr_lookup = f"{conf.behavior}[{true_eta}]"
    lr_method = f"{'cf' if len(conf.deployments) == 0 else 'hybrid'}-{conf.gradient}"

    # Start all computation
    xs = np.array([0, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000,
                   100_000, 150_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000, 550_000,
                   600_000, 650_000, 700_000, 750_000, 800_000, 850_000, 900_000, 950_000, 1_000_000])
    results = []
    for i in range(xs.shape[0]):
        sessions = xs[i]
        lr = learnrates[lr_lookup][f"{conf.cutoff}"][lr_method][f"{10_000 if sessions == 0 else sessions}"]
        results.append([
            evaluate(data, conf.baseline, conf.zero_based, conf.epochs, sessions, lr, conf.cutoff, conf.eps_1, conf.eps_2, conf.eta, conf.behavior, conf.gradient, list(filter(lambda d: d < sessions, conf.deployments)), seed_base + seed)
            for seed in range(repeats)
        ])
    
    # Await async results and compute statistics
    test_ys = []
    test_ys_std = []
    vali_ys = []
    vali_ys_std = []
    for i in range(xs.shape[0]):
        results[i] = [await r for r in results[i]]
        test_at_i = np.array([r['test']['ndcg@10'] for r in results[i]])
        vali_at_i = np.array([r['vali']['ndcg@10'] for r in results[i]])
        test_ys.append(np.mean(test_at_i))
        vali_ys.append(np.mean(vali_at_i))
        test_ys_std.append(np.std(test_at_i))
        vali_ys_std.append(np.std(vali_at_i))

    # Return results
    return {
        'args': vars(conf),
        'xs': xs,
        'test': {
            'ys': test_ys,
            'ys_std': test_ys_std,
        },
        'vali': {
            'ys': vali_ys,
            'ys_std': vali_ys_std
        },
        'n': repeats
    }


@task
async def evaluate(data, baseline, zero_based, epochs, sessions, learnrate, cutoff, eps_1, eps_2, eta, behavior, gradient, deployments, seed):
    vali_data, test_data = load_vali(data), load_test(data)
    w = train_model(data, baseline, zero_based, epochs, sessions, learnrate, cutoff, eps_1, eps_2, eta, behavior, gradient, deployments, seed)
    vali_data, test_data, w = await vali_data, await test_data, await w

    vali_performance = evaluate2(vali_data, w)
    test_performance = evaluate2(test_data, w)

    logging.info(f"[{seed:4d}, {sessions:7d}] vali: {vali_performance['ndcg@10']:.5f} test: {test_performance['ndcg@10']:.5f}")
    return {
        'test': vali_performance,
        'vali': test_performance
    }


@task(use_cache=True)
async def train_model(data, baseline, zero_based, epochs, sessions,
                         learnrate, cutoff, eps_1, eps_2, eta, behavior,
                         gradient, deployments, seed):

    # Random seed
    prng = rng_seed(seed)

    # Get data
    train_data = await load_train(data)
    
    # Load baseline(s)
    logging.info("Loading baseline model(s) ")
    w_0 = load_svmrank_model(baseline, zero_based)
    baselines = [(0, w_0)]
    for i in range(len(deployments)):
        logging.info(f"Getting model for deployment at {deployments[i]} sessions")
        bl_model = await train_model(data, baseline, zero_based, epochs,
                                     deployments[i], learnrate, cutoff, eps_1,
                                     eps_2, eta, behavior, gradient,
                                     deployments[:i], seed)
        baselines.append((deployments[i], bl_model))
    
    # Generate click data
    logging.info("Generating click data")
    indices = prng.randint(0, train_data.size, size=sessions)
    bcm = behaviors[behavior](cutoff, eta, eps_1, eps_2)
    rankings, clicks = run_click_simulation(train_data, indices, bcm, baselines)
    true_eta = 0.0 if behavior == "perfect" else eta

    # Train model
    logging.info("Training model")
    w = optimize_model(train_data, indices, rankings, clicks, gradient, epochs,
                       learnrate, p_eta=true_eta)
    
    # Evaluate model
    #logging.info("Evaluating results on held-out data")
    # return {
    #     "model": w,
    #     "test": evaluate2(datasets[data]['test'], w),
    #     "vali": evaluate2(datasets[data]['vali'], w),
    # }
    return w


if __name__ == "__main__":
    main()
