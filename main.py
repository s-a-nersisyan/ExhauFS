# External imports
import argparse
import os
import sys

# Internal imports

def dir(dir_name):
    dir_name = dir_name.strip('"')
    if not os.path.isdir(dir_name):
        raise ValueError(f"Directory {dir_name} does not exist.")
    return os.path.abspath(dir_name)

def file(file_name):
     file_name = file_name.strip('"')
     if not os.path.isfile(file_name):
        raise ValueError(f"File {file_name} does not exist.")
     return os.path.abspath(file_name)

class EFS(object):

    def __init__(self):
        # Create new parser of script arguments
        parser = argparse.ArgumentParser(
            usage='efs [-h] <command> [<args>]',
            description='''Exhaustive features selector

The most commonly used efs commands are:
  build       Build feature selectors
  estimate    Estimate running time
  summary     Get summary
''',
            formatter_class=argparse.RawDescriptionHelpFormatter)

        # Read the first positional argument defining a command
        parser.add_argument('command', metavar='command',
            type=str, choices=['build', 'estimate', 'summary'],
            help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        # Read arguments for a given command
        getattr(self, args.command)()

    def common_args(self, parser):
        parser.add_argument('-c', '--config', metavar='path',
            type=file, default="./config.json",
            help='Configuration file; Default: %(default)s.')
        
        # Example of bool option
        # parser.add_argument('--opt', action='store_true', help='Option description.')

    def build(self):
        # Create new parser for build arguments
        parser = argparse.ArgumentParser(
            prog='efs build',
            description='''Build feature selectors

The most commonly used building mode are:
  classifiers    Build tuples of features using predictive classification model
  regressors     Build tuples of features using predictive regression model
''',
            formatter_class=argparse.RawDescriptionHelpFormatter)

        # Add common options
        self.common_args(parser)

        # Add build options
        parser.add_argument('mode', metavar='mode',
            type=str, choices=['classifiers', 'regressors'],
            help='Building mode')

        # Parser build options
        args = parser.parse_args(sys.argv[2:])

        # Run builder
        if args.mode == "classifiers":
            import build_classifiers
            build_classifiers.main(args.config)

        elif args.mode == "regressors":
            import build_regressors
            build_regressors.main(args.config)

    def estimate(self):
        # Create new parser for estimate arguments
        parser = argparse.ArgumentParser(
            prog='efs estimate',
            description='''Estimate pipeline running time

The most commonly used esimating modes are:
  grid     Estimate pipeline running time for all values of n and k 
           on the grid of given size using data from a warm-up run 
           defined by values of warm_up_n and warm_up_k.
  max      Search the maximal number of selected features for which 
           estimated pipeline running time is less than max_estimated_time.
           Pipeline running time is calculated using bounded number of
           feature subsets.
  until    Calculate estimated pipeline running time using bounded
           number of feature subsets while pipeline running time 
           is less than max_estimated_time. 
''',
            formatter_class=argparse.RawDescriptionHelpFormatter)

        # Add common options
        self.common_args(parser)

        # Add estimate options
        parser.add_argument('mode', metavar='mode',
            type=str, choices=['grid', 'max', 'until'],
            help='Estimating mode')
        parser.add_argument('--grid_size', metavar='<num>', 
            type=int, default=100,
            help='Size of grid for values n and k; Default: %(default)s.')
        parser.add_argument('--warm_up_n', metavar='<num>', 
            type=int, default=50,
            help='Values of n for warm-up; Default: %(default)s.')
        parser.add_argument('--warm_up_k', metavar='<num>',
            type=int, default=2,
            help='Values of k for warm-up; Default: %(default)s.')
        parser.add_argument('--max_k', metavar='<num>', 
            type=int, default=100,
            help='Maximal length of features subset; Default: %(default)s.')
        parser.add_argument('--max_estimated_time', metavar='<time>', 
            type=float, default=24,
            help='Maximal estimated time of single pipeline running in hours; '\
                 'Default: %(default)s.')
        parser.add_argument('--n_feature_subsets', metavar='<num>', 
            type=int, default=100,
            help='Number of processed feature subsets; Default: %(default)s.')

        # Parser estimate options
        args = parser.parse_args(sys.argv[2:])

        # Run estimator
        if args.mode == "grid":
            import time_n_k
            time_n_k.main(
                args.config,
                args.warm_up_n,
                args.warm_up_k, 
                args.grid_size)

        elif args.mode == "max" or args.mode == "until":
            import running_time_estimator
            running_time_estimator.main(
                args.config, 
                args.max_k, 
                args.max_estimated_time,
                args.n_feature_subsets, 
                args.mode == "max")

    def summary(self):
        # Create new parser for summary arguments
        parser = argparse.ArgumentParser(
            prog='efs summary',
            description='''Get summary

The most commonly used summary modes are:
  classifiers    Fit and evaluate classifier, generate report of its accuracy 
                 metrics for datasets, plot its features importances (only for 
                 SVM classifier), and save trained model to file. 
  features       Read classifiers.csv from output directory and generate table 
                 that for each feature contains percentage of reliable classifiers 
                 which use it.
''',
            formatter_class=argparse.RawDescriptionHelpFormatter)

        # Add common options
        self.common_args(parser)

        # Add summary options
        parser.add_argument('mode', metavar='mode',
            type=str, choices=['classifiers', 'features'],
            help='Summary mode')
        parser.add_argument('--output_dir', metavar='<dir>', 
            type=dir, default=os.path.abspath(os.getcwd()),
            help='Directory with output files; Default: %(default)s.')
        parser.add_argument('--plot', metavar='<file>', 
            type=file,
            help='Plot classifier-participation-curves for features.')

        # Parser summary options
        args = parser.parse_args(sys.argv[2:])

        # Get summary
        if args.mode == "classifiers":
            import make_classifier_summary
            make_classifier_summary.main(args.config)

        elif args.mode == "features":

            # Gererate table that for each feature contains percentage 
            # of reliable classifiers which use it.
            import get_features_summary
            get_features_summary.main(args.output_dir)

            # Plot classifier-participation-curves for features
            if args.plot is not None:
                import plot_features_summary
                plot_features_summary.main(args.plot)


if __name__ == '__main__':
    EFS()
