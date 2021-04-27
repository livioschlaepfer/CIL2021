import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_false")
args = parser.parse_args()
print(args)
if args.verbose:
    print("verbosity turned on")