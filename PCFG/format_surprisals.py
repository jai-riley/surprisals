import pandas as pd


def read_roark_surprisals(path):
    with open(path, 'r') as file:
        line = file.readline()
        tokens = []
        surp = []
        syn_surp = []
        lex_surp = []
        while line:
            if line[0:5] == 'pfix:' or line[0:5] == 'pfix-':
                values = line.split()
                tokens.append(values[1])
                surp.append(values[3])
                syn_surp.append(values[4])
                lex_surp.append(values[5])
            line = file.readline()
    surprisals = pd.DataFrame(data={'token': tokens, 'surp': surp, 'syn_surp': syn_surp, 'lex_surp': lex_surp})
    return surprisals


def main():
    for data_type in ["word", "pos"]:
        for stimuli in ["L2"]:
            data_path = f'/Users/jairiley/Desktop/BOW_Ngrams/PCFG/incremental-top-down-parser/model/wiki_{data_type}_02.output'
            surprisals = read_roark_surprisals(data_path)
            surprisals = surprisals[surprisals.token != '</s>']
            if data_type == "word":
                for surp in ["surp", "syn_surp", "lex_surp"]:
                    output_path = f'/Users/jairiley/Desktop/BOW_Ngrams/PCFG/incremental-top-down-parser/model/wiki_{data_type}_{surp}.csv'
                    surprisals.to_csv(output_path, columns=["token", surp], index=False, header=False)
            else:
                output_path = f'/Users/jairiley/Desktop/BOW_Ngrams/PCFG/incremental-top-down-parser/model/wiki_{data_type}_surp.csv'
                surprisals.to_csv(output_path, columns=["token", "surp"], index=False, header=False)


if __name__ == "__main__":
    main()
