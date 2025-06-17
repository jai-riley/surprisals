import kenlm
import math
import sys
import nltk
import json
import pandas as pd
import argparse


def read_lines(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
    return lines


def find_n_grams(lines, n):
    grams = []
    for line in lines:
        line_grams = list(
            nltk.ngrams(line.strip("\n").split(), n, pad_right=True, right_pad_symbol="</s>", pad_left=True,
                        left_pad_symbol="<s>"))
        grams.extend(line_grams)
    return grams


def compute_surprisals(grams, model):
    surprisals = []
    for gram in grams:
        string = " ".join(gram)
        # print(list(model.full_scores(string)))
        log_prob = list(model.full_scores(string))[-2][0]
        # surprisal = [gram[-1], -log_prob / math.log(math.e,10)] 
        surprisal = [gram[-1], -log_prob * math.log(10,math.e)]  # base-10 → natural log

        surprisals.append(surprisal)
    return surprisals


def main(args):
    lines = read_lines(args.target)
    grams = find_n_grams(lines, args.n)
    n = args.n
    if n == 6:
        model = kenlm.Model(f"/Users/jairiley/Desktop/BOW_Ngrams/Ngram/models/wiki_{args.type}_6.arpa")
    elif n == 5:
        model = kenlm.Model(f"/Users/jairiley/Desktop/BOW_Ngrams/Ngram/models/wiki_{args.type}_5.arpa")
    elif n == 4:
        model = kenlm.Model(f"/Users/jairiley/Desktop/BOW_Ngrams/Ngram/models/wiki_{args.type}_4.arpa")
    elif n == 3:
        model = kenlm.Model(f"/Users/jairiley/Desktop/BOW_Ngrams/Ngram/models/wiki_{args.type}_3.arpa")
    else:
        model = kenlm.Model(f"/Users/jairiley/Desktop/BOW_Ngrams/Ngram/models/wiki_{args.type}_2.arpa")
    surprisals = compute_surprisals(grams, model)
    surprisals = pd.DataFrame(surprisals)
    surprisals.to_csv(args.savefile,index=False,header=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments")

    # Adding arguments
    parser.add_argument("--type", type=str, help="{pos, word, word_pos}", required=True)
    parser.add_argument("--n", type=int, help="Value of n", required=True)
    parser.add_argument("--target", type=str, help="Target file", required=True)
    parser.add_argument("--savefile", type=str, help="File to save the output", required=True)

    # Parsing arguments
    args = parser.parse_args()

    main(args)
