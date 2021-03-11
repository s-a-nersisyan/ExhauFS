# External imports
import argparse
import sys

# Internal imports
import build_classifiers
import estimator

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
                            type=str, choices=['build','estimate','summary'],
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
            prog = 'efs build',
            description='Build classifiers')

        # Add common options
        self.common_args(parser)

        # Add build options

        # Parser build options
        args = parser.parse_args(sys.argv[2:])

        # Run builder
        build_classifiers.main(args.config)

    def estimate(self):
        
        # Create new parser for estimate arguments
        parser = argparse.ArgumentParser(
            prog = 'efs estimate',
            description='Estimate running time')

        # Add common options
        self.common_args(parser)

        # Add estimate options

        # Parser estimate options
        args = parser.parse_args(sys.argv[2:])

        # Run estimator
        estimator.main(args.config)

    def summary(self):
        
        # Create new parser for summary arguments
        parser = argparse.ArgumentParser(
            prog = 'efs summary',
            description='Get summary')

        # Add common options
        self.common_args(parser)

        # Add summary options

        # Parser summary options
        args = parser.parse_args(sys.argv[2:])

        # Get summary


if __name__ == '__main__':
    EFS()