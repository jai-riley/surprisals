import json
from pathlib import Path
import re
import pandas as pd


def get_output(output_path, dataset):
    result_time_files = output_path.glob(dataset + '*.txt')
    result_files = output_path.glob('*.json')
    results = pd.DataFrame(
        columns=["lr_scheduler", "lr", "dropout_rate", "batch_size", "batch_group", "val_loss", "val_word_ppl",
                 "execution_time"])
    count = 0
    for file in result_time_files:
        count += 1
        print(count)
        pattern = dataset + '_pos_time_(.+)_(.+)_(.+)_(.+)_(similar_length|random)'
        parameters = re.search(pattern, file.stem).groups()
        with open(file) as f:
            output = f.read().split('\n')
            print(output)
            time_line = output[len(output) - 4].split()
            time = time_line[1]
            print(time)
            print(file.stem)
            minutes = time.split('m')[0]
            seconds = time.split('m')[1][0:-1]
            total_time = int(minutes) * 60 + float(seconds)

        results = pd.concat([results, pd.DataFrame({"lr_scheduler": [parameters[0]],
                                                    "lr": [parameters[1]],
                                                    "dropout_rate": [parameters[2]],
                                                    "batch_size": [parameters[3]],
                                                    "batch_group": [parameters[4]],
                                                    "val_loss": [0],
                                                    "val_word_ppl": [0],
                                                    "execution_time": [total_time]})], ignore_index=True)

    for file in result_files:
        pattern = 'val_word_ppls_(.+)_(.+)_(.+)_(.+)_(similar_length|random)'
        parameters = re.search(pattern, file.stem).groups()
        parameters = list(parameters)
        if parameters[2] == '0.0':
            parameters[2] = '0'
        index = results.index[(results['lr_scheduler'] == parameters[0]) & (results['lr'] == parameters[1]) & (
                    results['dropout_rate'] == parameters[2]) & (results['batch_size'] == parameters[3]) & (
                                          results['batch_group'] == parameters[4])]
        with open(file) as f:
            val_performance = json.load(f)
        val_performance_dict = {val_performance['val_losses'][i]: val_performance['val_word_ppls'][i] for i in
                                range(len(val_performance['val_losses']))}
        val_loss = min(val_performance['val_losses'])
        val_word_ppl = val_performance_dict[val_loss]
        results.loc[index, ['val_loss']] = val_loss
        results.loc[index, ['val_word_ppl']] = val_word_ppl

    return results


def main():
    dataset = 'wiki'
    for tuning_round in range(1,3):
        output_path = Path(f'../../../output/RNNG/pos/hyperparameter_tuning/tuning_round_{tuning_round}/')
        new_output_path = Path(f'../../../output/RNNG/pos/hyperparameter_tuning/round_{tuning_round}_wiki_parameter_results.csv')
        results = get_output(output_path, dataset)
        results.to_csv(new_output_path, index=False)


if __name__ == "__main__":
    main()
