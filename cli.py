"""cli.py

A simple cli for running simulations.

Usage:

As the user, there are still many options you can tweak by editing this file 
directly. However, we set up this simple CLI to run a basic LiRA.
"""
from topic_models import train_lda_sklearn
from model_stats import max_lda_logll
from simulation_utils import train_target_model, train_shadow_models, fit_lira

import argparse

parser = argparse.ArgumentParser(prog= 'LiRA Simulation',
                    description= 'This program simulates a LiRA')

parser.add_argument("-d", help='Path to data')
parser.add_argument("-k", type=int, help="Number of topics")
parser.add_argument("-N", default=128, type=int, help = "Number of shadow models (Default 128)")
parser.add_argument("-o", default= "./lira/", help="Output path (Default ./lira/)")

stat_funcs = [max_lda_logll]

if __name__ == "__main__":
    args = parser.parse_args()
    train_target_model(args.d, args.o, .5, train_lda_sklearn, {"k": args.k})
    print()
    train_shadow_models(args.d, args.o, args.o, args.N, train_lda_sklearn, {"k": args.k})
    print()
    fit_kwargs = {'out_path': args.o, 'statistic_functions': stat_funcs}
    fit_lira(args.d, args.o, args.o, fit_kwargs)