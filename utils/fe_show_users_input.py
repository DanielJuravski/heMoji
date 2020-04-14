import json

RECORDS_PATH = '/home/daniel/Downloads/emoji_predictor.log'
OUTPUT_PATH = '/home/daniel/heMoji/logs/streamlit_parsed_records.txt'
TOP_E = 5


def load_record():
    data = []
    with open(RECORDS_PATH, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line_dict = json.loads(line)
        sample = {}
        sample['input'] = line_dict['input']
        sample['emoji'] = line_dict['prediction']['emoji'][:TOP_E]
        data.append(sample)

    return data


def dump_to_file(data):
    with open(OUTPUT_PATH, 'w') as f:
        for sample in data:
            input = sample['input'].encode('utf-8')
            emoji = "".join([e.encode('utf-8') for e in sample['emoji']])
            f.writelines("{0}: {1}".format(input, emoji))
            f.writelines("\n")


if __name__ == '__main__':
    """
    load emoji_predict.log file, write to file input sentence and it's prediction
    """
    data = load_record()
    dump_to_file(data)
    print("Done!")
