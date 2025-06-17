import kenlm
import math
import sys
import nltk
import json
import pandas as pd
import argparse
import time
from itertools import permutations
first = True

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
        line_grams = line_grams[:-(n-1)]
        grams.extend(line_grams)
    return grams


def compute_surprisals(grams, n):
    surprisals = []
    for gram in grams:
        # print(gram)
        log_prob = compute_log_prob(gram, n)
        surprisal = [gram[-1], -log_prob / math.log(math.e, 10)]  # convert to log base 2
        surprisals.append(surprisal)
    return surprisals


def compute_log_prob(gram, n):
    uperms = get_uperms(gram)
    numerator = get_numerator(uperms, n)  # numerator is in log percent space
    denominator = get_denominator(uperms, n)  # denominator is in log percent space
    log_prob = numerator - denominator
    # first = False
    return log_prob


def get_uperms(gram):
    wn = gram[-1]
    uperms = []
    tails = list(set(permutations(gram[:-1])))
    for tail in tails:
        uperms.append(" ".join(tail) + " " + wn)
    return uperms


def get_numerator(uperms, n):
    numerator = 0  # in percent space not log space
    for uperm in uperms:
        numerator += get_numerator_term(uperm, n)  # value in percent space
    # if first:
    #     print()
    return math.log(numerator, 10)  # return in log percent space


def get_numerator_term(uperm, n):
    # return in percent space (probability space * 100)
    num_term = 0
    if n == 6:

        num_term += list(six_gram_model.full_scores(uperm))[-2][0]
        uperm = uperm.rsplit(" ", 1)[0]
    if n >= 5:
        num_term += list(five_gram_model.full_scores(uperm))[-2][0]
        uperm = uperm.rsplit(" ", 1)[0]
    if n >= 4:
        num_term += list(four_gram_model.full_scores(uperm))[-2][0]
        uperm = uperm.rsplit(" ", 1)[0]
    if n >= 3:
        # print(uperm)
        num_term += list(three_gram_model.full_scores(uperm))[-2][0]
        uperm = uperm.rsplit(" ", 1)[0]
        # print(uperm)

    if n >= 2:
        num_term += list(two_gram_model.full_scores(uperm))[-2][0]
        uperm = uperm.rsplit(" ", 1)[0]
        try:
            num_term += (math.log(one_gram_dict["freq_dict"][uperm], 10) - math.log(one_gram_dict["total"], 10))
        except:
            # pass
            num_term += (math.log(one_gram_dict["freq_dict"]['<unk>'], 10) - math.log(one_gram_dict["total"], 10))

    return 10 ** num_term


def get_denominator(uperms, n):
    denominator = 0  # in percent space not log space
    for uperm in uperms:
        denominator += get_denominator_term(uperm, n)  # value in percent space

    return math.log(denominator, 10)  # return in log percent space


def get_denominator_term(uperm, n):
    # return in percent space (probability space * 100)
    denom_term = 0
    # print(uperm)
    if n == 6:
        uperm = uperm.rsplit(" ", 1)[0]
        denom_term += list(five_gram_model.full_scores(uperm))[-2][0]
    if n >= 5:
        uperm = uperm.rsplit(" ", 1)[0]
        denom_term += list(four_gram_model.full_scores(uperm))[-2][0]
    if n >= 4:
        uperm = uperm.rsplit(" ", 1)[0]
        denom_term += list(three_gram_model.full_scores(uperm))[-2][0]
    if n >= 3:
        uperm = uperm.rsplit(" ", 1)[0]
        denom_term += list(two_gram_model.full_scores(uperm))[-2][0]
    if n >= 2:
        uperm = uperm.rsplit(" ", 1)[0]
        try:
            denom_term += (math.log(one_gram_dict["freq_dict"][uperm], 10) - math.log(one_gram_dict["total"], 10))
        except:
            # pass
            denom_term += (math.log(one_gram_dict["freq_dict"]["<unk>"], 10) - math.log(one_gram_dict["total"], 10))

    # conver denom_term to percent space 
    return 10 ** denom_term


def main(args):
    lines = read_lines(args.target)
    grams = find_n_grams(lines, args.n)
    surprisals = compute_surprisals(grams, args.n)
    surprisals = pd.DataFrame(surprisals)
    # print(surprisals.head(50))
    surprisals.to_csv(args.savefile,index=False)

    # print(surprisals)


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Process command line arguments")

    # Adding arguments
    parser.add_argument("--type", type=str, help="{pos, words, words_pos}", required=True)
    parser.add_argument("--n", type=int, help="Value of n", required=True)
    parser.add_argument("--target", type=str, help="Target file", required=True)
    parser.add_argument("--savefile", type=str, help="File to save the output", required=True)
    # first = True
    # Parsing arguments
    args = parser.parse_args()
    if int(args.n) == 6:
        six_gram_model = kenlm.Model(f"models/wiki_{args.type}_6.arpa")
    if int(args.n) >= 5:
        five_gram_model = kenlm.Model(f"models/wiki_{args.type}_5.arpa")
    if int(args.n) >= 4:
        four_gram_model = kenlm.Model(f"models/wiki_{args.type}_4.arpa")
    if int(args.n) >= 3:
        three_gram_model = kenlm.Model(f"models/wiki_{args.type}_3.arpa")
    if int(args.n) >= 2:
        two_gram_model = kenlm.Model(f"models/wiki_{args.type}_2.arpa")
        with open(f"models/wiki_{args.type}_1.json", "r") as f:
            one_gram_dict = json.load(f)

    main(args)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"{total_time//60}:{total_time%60}")