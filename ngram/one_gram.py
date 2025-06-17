import json
import os
import sys


if __name__ == "__main__":
    train_name = sys.argv[1]
    model_name = sys.argv[2]
    frequency_dict = {}
    total = 0
    with open(train_name, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        words = line.split()
        for word in words:
            if word in frequency_dict:
                frequency_dict[word] = frequency_dict[word] + 1
            else:
                frequency_dict[word] = 1
            total += 1
    frequency_dict["<s>"] = len(lines)
    frequency_dict["</s>"] = len(lines)
    frequency_dict["<unk>"] = 1
    total += 1
    final_dict = {"freq_dict": frequency_dict, "total": total}
    with open(model_name, 'w') as f:
        json.dump(final_dict, f)
