import csv

from dist.process.add_hemojis import get_emojis_keys


def iterate_emojis(emojis, emojis_count):
    for e in emojis.split():
        emojis_count[e.decode('utf-8')] += 1

    return emojis_count


def main():
    emojis = get_emojis_keys()
    t_emojis_count = {e: 0 for e in emojis}
    c_emojis_count = {e: 0 for e in emojis}

    with open('data.txt', 'r') as f:
        lines = f.readlines()

    for line in lines:
        t_emojis, c_emojis = line.split('\t')
        iterate_emojis(t_emojis, t_emojis_count)
        iterate_emojis(c_emojis, c_emojis_count)

    with open('data_count.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow([e.encode('utf-8') for e in t_emojis_count.keys()])
        wr.writerow(t_emojis_count.values())
        wr.writerow(c_emojis_count.values())


if __name__ == '__main__':
    """
    This script analyze the data that was created via 'create_data.py'
    """

    main()
