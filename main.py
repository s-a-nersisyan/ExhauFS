# External imports
import argparse
import sys

# Internal imports

class EFS(object):

    def __init__(self):
        # Create new parser of script arguments
        parser = argparse.ArgumentParser(
            usage='efs [-h] <command> [<args>]',
            description='''Exhaustive features selector

The most commonly used efs commands are:
  build       Build classifiers
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
                            type=str, default="./config.json",
                            help='Configuration file; Default: %(default)s')

    def build(self):
        # Create new parser for build arguments
        parser = argparse.ArgumentParser(
            prog='efs build',
            description='Build classifiers')

        # Add common options
        self.common_args(parser)

        # Add build options

        # Parser build options
        args = parser.parse_args(sys.argv[2:])

        # Run builder
        import build_classifiers
        build_classifiers.main(args.config)

    def estimate(self):
        # Create new parser for estimate arguments
        parser = argparse.ArgumentParser(
            prog='efs estimate',
            description='Estimate running time')

        # Add common options
        self.common_args(parser)

        # Add estimate options
        parser.add_argument('max_k', type=int,
                            help='Maximal length of features subset.')
        parser.add_argument('max_estimated_time', type=float,
                            help='Maximal estimated time of ' \
                                 'single pipeline running in hours.')
        parser.add_argument('n_feature_subsets', type=int,
                            help='Number of processed feature subsets ' \
                                 '(100 is pretty good).')
        parser.add_argument('--search_max_n', action='store_true',
                            help='Search max n for which estimated run time of ' \
                                 'the pipeline is less than max_estimated_time.')

        # Parser estimate options
        args = parser.parse_args(sys.argv[2:])

        # Run estimator
        import running_time_estimator
        running_time_estimator.main(args.config, args.max_k, args.max_estimated_time,
            args.n_feature_subsets, args.search_max_n)

    def summary(self):
        # Create new parser for summary arguments
        parser = argparse.ArgumentParser(
            prog='efs summary',
            description='Get summary')

        # Add common options
        self.common_args(parser)

        # Add summary options

        # Parser summary options
        args = parser.parse_args(sys.argv[2:])

        # Get summary


if __name__ == '__main__':
    EFS()
