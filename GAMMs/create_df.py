import pandas as pd
from get_log_freq import get_wiki_frequencies
import numpy as np
M = 2

def process_words(reading_times):
    words = []
    for index, row in reading_times.iterrows():
        if type(row["Word"]) != str:
            print(row)
        # if str(row["Word"]) != "nan":

        words.append(row["Word"].replace(".", "").replace(",", "").lower())
        # else:
        #     words.append("")
    reading_times["procWordID"] = words
    return reading_times


def get_reading_times():
    reading_times_path = '/Users/jairiley/Desktop/BOW_Ngrams/corpus/All_Groups_with_accuracy_and_RT.xlsx'
    sheets_info = {'L1ReadL2materials': ['L2', 'E'],
                   'Chinese-English Speakers': ['L2', 'C'],
                   'Spanish-English Speakers': ['L2', 'S'],
                   'korean_sorted': ['L2', 'K']}
    reading_times = pd.DataFrame()
    for sheet, info in sheets_info.items():
        sheet = pd.read_excel(reading_times_path, sheet_name=sheet)
        sheet['Stimuli'] = info[0]
        sheet['Reader'] = info[1]
        reading_times = pd.concat([reading_times, sheet], ignore_index=True)

    return reading_times


def get_processed_word(reading_times, type, label):
    data_path = f"../corpus/L2/stimuli_L2_{type}.txt"
    sent = 0
    token_list = []
    token_line = []
    with open(data_path) as file:
        for line in file:
            for word in line.strip().split():
                if word != "," and word != ".":
                    token_line.append(word)

            sent += 1
            if sent == 2:
                token_list.append(token_line)
                token_line = []
                sent = 0

    tokens = []
    for _, row in reading_times.iterrows():
        tokens.append(token_list[int(row["ItemNum"]) - 1][int(row["WordNo"]) - 1])

    reading_times[label] = tokens
    return reading_times


def get_pos_tags(reading_times):
    POS = "../corpus/L2/stimuli_L2_pos.txt"
    sent = 0
    pos_tag_list = []
    pos_line = []
    with open(POS) as file:
        for line in file:
            for word in line.strip().split():
                if word != "," and word != ".":
                    pos_line.append(word)

            sent += 1
            if sent == 2:
                pos_tag_list.append(pos_line)
                pos_line = []
                sent = 0

    pos_tags = []
    for _, row in reading_times.iterrows():
        pos_tags.append(pos_tag_list[int(row["ItemNum"]) - 1][int(row["WordNo"]) - 1])

    reading_times["POS"] = pos_tags
    return reading_times


def get_proficiencies(reading_times):
    prof_path = '/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/Language Proficiency and Residence in the US.xlsx'
    sheets_info = {'L2': True,
                   'L1 Reads L2': False}
    subject_info = {}
    for sheet, residency in sheets_info.items():
        sheet = pd.read_excel(prof_path, sheet_name=sheet)
        for subject in sheet["LID"].unique():
            info = []
            for column in ["Comp_Performance%Acc", "Comp_Competence%Acc", "Vocab_Performance%Acc",
                           "Vocab_Competence%Acc"]:
                info.append(sheet.loc[sheet["LID"] == subject][column].values[0])
            if residency:
                info.append(sheet.loc[sheet["LID"] == subject]["LOR_month"].values[0])

            else:
                info.append(-1)

            subject_info[subject] = info

    comp_perf = []
    comp_comp = []
    vocab_perf = []
    vocab_comp = []
    months = []

    for index, row in reading_times.iterrows():
        subject = row["Subject"]
        comp_perf.append(subject_info[subject][0])
        comp_comp.append(subject_info[subject][1])
        vocab_perf.append(subject_info[subject][2])
        vocab_comp.append(subject_info[subject][3])
        months.append(subject_info[subject][4])

    reading_times["Comp_Performance%Acc"] = comp_perf
    reading_times["Comp_Competence%Acc"] = comp_comp
    reading_times["Vocab_Performance%Acc"] = vocab_perf
    reading_times["Vocab_Competence%Acc"] = vocab_comp
    reading_times["LOR_month"] = months

    return reading_times


def get_trials(reading_times):
    trials = []
    trial_count = 1
    prev_ItemNum = -1
    prev_subject = "-1"
    for index, row in reading_times.iterrows():
        if prev_subject != row["Subject"]:
            prev_subject = row["Subject"]
            prev_ItemNum = row["ItemNum"]
            trial_count = 1

        elif prev_ItemNum != row["ItemNum"]:
            prev_subject = row["Subject"]
            prev_ItemNum = row["ItemNum"]
            trial_count += 1

        trials.append(trial_count)

    reading_times["Trial"] = trials
    return reading_times


def get_word_lengths(reading_times):
    # word_lengths = []
    # for word in reading_times['Word']:
    #     if str(word) != "nan":
    #         if "." not in str(word) and "," not in str(word):
    #             word_lengths.append(len(word))
    #         else:
    #             word_lengths.append(len(word)-1)
    #     else:
    #         word_lengths.append(0)

    word_lengths = [len(word) if ("." not in str(word) and "," not in str(word)) else len(word) - 1 for word in
                    reading_times['Word']]

    reading_times['WordLength'] = word_lengths
    for i in range(1, M):
        reading_times[f'WordLengthPrev{i}'] = reading_times['WordLength'].shift(i)
    return reading_times


def get_log_frequencies(reading_times):
    log_frequencies = get_wiki_frequencies("train")
    log_frequencies = [log_frequencies[word] for word in reading_times['procWord']]
    reading_times['LogFreq'] = log_frequencies
    for i in range(1, M):
        reading_times[f'LogFreqPrev{i}'] = reading_times['LogFreq'].shift(i)
    return reading_times


def get_word_positions(reading_times):
    word_positions = []
    sentence_positions = []
    num_sentences = 0
    word_count = 0
    for index, word_no in reading_times['WordNo'].items():
        word_count += 1
        word_positions.append(word_count)
        if num_sentences == 2:
            num_sentences = 0
        sentence_positions.append(num_sentences + 1)
        if "." in str(reading_times['Word'][index]):
            num_sentences += 1
            word_count = 0

    reading_times['WordPos'] = word_positions
    reading_times['SentPos'] = sentence_positions

    return reading_times


def get_rows_included(reading_times):
    for i in range(1, M):
        rows_included = [False if word_pos <= i else True for _, word_pos in reading_times['WordPos'].items()]
        reading_times[f'IncludePrev{i}'] = rows_included
    return reading_times


def get_has_punct(reading_times):
    has_punct = [1 if "." in str(word) or "," in str(word) else 0 for _, word in reading_times['Word'].items()]
    reading_times['HasPunct'] = has_punct
    return reading_times


def get_surprisals(path, stimuli):
    surprisal = pd.read_csv(path, header=None, names=['token', 'surp'])
    surprisal = surprisal.dropna()
    surprisal = surprisal[surprisal.token != "</s>"]

    item_nums = []
    word_nos = []
    item_count = 1
    word_count = 0
    num_sentences = 0
    for token in surprisal['token']:
        if path == '../surprisals/stimuli_word_pos_6.csv' or path == '/Users/jairiley/Desktop/BOW_Ngrams/surprisals/stimuli_pos_word_BOW.csv' or path == '../surprisals/trans_L2_word_pos_surp.csv':
            if not (token == ",/," or token == "./."):
                word_count += 1
                word_nos.append(word_count)
                item_nums.append(item_count)
            if token == "./.":
                num_sentences += 1
            if num_sentences == 2:
                item_count += 1
                word_count = 0
                num_sentences = 0
        else:
            if not (token == "," or token == "."):
                word_count += 1
                word_nos.append(word_count)
                item_nums.append(item_count)
            if token == ".":
                num_sentences += 1
            if num_sentences == 2:
                item_count += 1
                word_count = 0
                num_sentences = 0
    if path == '../surprisals/stimuli_word_pos_6.csv' or path == '/Users/jairiley/Desktop/BOW_Ngrams/surprisals/stimuli_pos_word_BOW.csv' or path == '../surprisals/trans_L2_word_pos_surp.csv':
        surprisal = surprisal[surprisal.token != "./."]
        surprisal = surprisal[surprisal.token != ",/,"]
    else:
        surprisal = surprisal[surprisal.token != "."]
        surprisal = surprisal[surprisal.token != ","]
    surprisal['WordNo'] = word_nos
    surprisal['ItemNum'] = item_nums
    surprisal['Stimuli'] = stimuli

    return surprisal


def combine_surprisal_reading_times(rt, surprisal, column_name):
    for index, row in surprisal.iterrows():
        # if type(row["surp"]) != float:
        #     print(column_name)
        #     print(row["surp"])
        rt.loc[(rt["Stimuli"] == row["Stimuli"])
               & (rt["ItemNum"] == row["ItemNum"])
               & (rt["WordNo"] == row["WordNo"]), [column_name]] = float(row["surp"])

    for i in range(1, M):
        rt[f'{column_name}Prev{i}'] = rt[column_name].shift(i)
    return rt

def remove_prev(df):
    prev_columns = [col for col in df.columns if 'prev' in col.lower() and col.lower() != 'includeprev1']

    # Iterate over 'prev' columns and update them to NaN where 'IncludePrev1' is False
    for col in prev_columns:
        df.loc[df['IncludePrev1'] == False, col] = df[col].mean()
    
    return df

def main():
    surprisal_paths = {'n_gram_word': '../surprisals/stimuli_word_6.csv',
                       'n_gram_POS': '../surprisals/stimuli_pos_5.csv',
                       'n_gram_word_POS': '../surprisals/stimuli_word_pos_6.csv',
                       'n_gram_BOW_word': '../surprisals/stimuli_word_BOW_6.csv',
                       'n_gram_BOW_POS': '../surprisals/stimuli_pos_BOW.csv',
                       'n_gram_BOW_word_POS': '/Users/jairiley/Desktop/BOW_Ngrams/surprisals/stimuli_pos_word_BOW.csv',
                       'PCFG_pos': '../surprisals/PCFG_pos_surp.csv',
                       'PCFG_total': '../surprisals/PCFG_word_surp.csv',
                       'PCFG_lex': '../surprisals/PCFG_word_lex_surp.csv',
                       'PCFG_syn': '../surprisals/PCFG_word_syn_surp.csv',
                       'RNNG_word': '../surprisals/stimuli_RNNG_word.csv',
                       'RNNG_pos': '../surprisals/RNNG_stimuli_pos.csv',
                        'transformer_word': '../surprisals/trans_L2_word_surp.csv',
                       'transformer_pos': '../surprisals/trans_L2_pos_surp.csv',
                       'transformer_word_pos': '../surprisals/trans_L2_word_pos_surp.csv',
                       'TG_word': '../surprisals/TG_word_surp_30000.csv',
                       }

    reading_times = get_reading_times()
    print("Reading times loaded")

    reading_times = get_processed_word(reading_times, "word", "procWord")
    print("procWord found")

    reading_times = process_words(reading_times)
    print('procWordID found')

    reading_times = get_pos_tags(reading_times)
    print("POS tags found")

    reading_times = get_proficiencies(reading_times)
    print("Proficiencies found")

    reading_times = get_trials(reading_times)
    print("Trials found")

    reading_times = get_has_punct(reading_times)
    print("Punctuation found")

    reading_times = get_word_positions(reading_times)
    print("Word positions found")

    reading_times = get_rows_included(reading_times)
    print("Rows included found")

    reading_times = get_word_lengths(reading_times)
    print("Word lengths found")

    reading_times = get_log_frequencies(reading_times)
    print("Log frequencies found")

    for model, paths in surprisal_paths.items():
        L2_surp = get_surprisals(paths, 'L2')
   
        reading_times = combine_surprisal_reading_times(reading_times, L2_surp, f'{model}_surp')
        # if model == "n_gram_BOW_word_POS":
        #     print(reading_times["n_gram_BOW_word_POS"])
        print(f'{model} done')

    reading_times = remove_prev(reading_times)
    print("removed prev")
    reading_times.to_csv("all_data.csv", index=False)


if __name__ == "__main__":
    main()
