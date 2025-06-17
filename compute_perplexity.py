import math
import pandas as pd
import argparse
import numpy as np



def main(args):
    surprisals = pd.read_csv(args.surprisal_file,header=None)
    negative_log_prob = 0 # log base e
    count = 0
    for ind in surprisals.index:
        # print(surprisals)
        if surprisals[0][ind] != "<s>" and surprisals[0][ind] != "</s>" and str(surprisals[0][ind]) != "nan":
            #and str(surprisals[0][ind]) != ".": for TG
            # print(surprisals[0][ind]
            # print(surprisals[1][ind])
            count += 1
            negative_log_prob += float(surprisals[1][ind])
    print(count)
    print(f"neg log prob: {negative_log_prob}")
    print(f"average neg log prob: {negative_log_prob/count}")
    print(f"Perplexity: {math.e ** (negative_log_prob/count)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments")
    
    # Adding arguments
    parser.add_argument("--surprisal_file", type=str, help="{pos, word, word_pos}", required=True)
    
    # Parsing arguments
    args = parser.parse_args()
    
    main(args)
