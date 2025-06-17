import pandas as pd
import math


def main():
    stimuli = ['L1', 'L2', 'test', 'validation']
    for stimulus in stimuli:
        surprisal_path = f'../../../output/RNNG/pos/surprisals/surprisals_{stimulus}_pos.txt'
        surprisals = pd.read_csv(surprisal_path, sep='\t',
                                 names=['sent_index', 'token_index', 'token', 'mod_token', 'word_surp',
                                        'piece_surp'], low_memory=False)
        output_path = f'../../../output/RNNG/pos/formatted_surprisals/stimuli_{stimulus}_RNNG_pos_surp.csv'
        perplexity = math.exp(surprisals['word_surp'].sum()/surprisals['word_surp'].count())
        print(stimulus, perplexity)
        surprisals.to_csv(output_path, columns=['token', 'word_surp'], header=False, index=False)


if __name__ == '__main__':
    main()
