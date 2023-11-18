from pmlm.log import init_logging
init_logging()
from pmlm.parser.main import parse_args

def main():
    args = parse_args()
    args.handler(args)

