import json
import sys

ORIG_RECORD_FILE = '../data/hebrew_tweets_wtime_3G.json'
TWEETS_META_CUSTOM_LEN_FILE = '../data/hebrew_tweets_wtime_3G_mini.json'
TWEETS_TEXT_CUSTOM_LEN_FILE = '../data/hebrew_tweets_3G_mini.tsv'
LINES_TO_PROCESS = 1000


def getArgs():
    if len(sys.argv) == 5:
        orig_file = sys.argv[1]
        tweets_meta_file = sys.argv[2]
        tweets_text_file = sys.argv[3]
        lines_to_process = sys.argv[4]
    else:
        orig_file = ORIG_RECORD_FILE
        tweets_meta_file = TWEETS_META_CUSTOM_LEN_FILE
        tweets_text_file = TWEETS_TEXT_CUSTOM_LEN_FILE
        lines_to_process = LINES_TO_PROCESS

    print("\nReading original file: {0}\n\nWriting {1} lines of tweets meta to: {2}\n\nWriting the text out of these meta to: {3}\n".format(
        orig_file, lines_to_process, tweets_meta_file, tweets_text_file))

    return orig_file, tweets_meta_file, tweets_text_file, lines_to_process


def extractCustomLengthTweets(orig_file, tweets_meta_file, lines_to_process):
    # initialize (clean) file
    with open(tweets_meta_file, 'w') as f_mini:
        f_mini.write("")

    # print (lines_to_process) lines from (orig_file) to (tweets_meta_file)
    with open(orig_file, 'r') as f:
        for i in range(lines_to_process):
            x = f.readline()
            if i % 10000 == 0:
                print("Reading line {} out of {}".format(i, lines_to_process))
            with open(tweets_meta_file, 'a') as f_mini:
                f_mini.write(x)


def extractTextFromTweets(tweets_meta_file, tweets_text_file):
    # initialize (clean) file
    with open(tweets_text_file, 'w') as f_text:
        f_text.write("")

    with open(tweets_text_file, 'a') as f_text:
        # tsv_writer = csv.writer(f_tsv, delimiter='\t')
        with open(tweets_meta_file, 'r') as f_meta:
            while True:
                x = f_meta.readline()
                if not x: break
                if x != '\n':
                    js = json.loads(x)
                    if 'text' in js:
                        f_text.write(js['text'].encode('utf-8'))
                        f_text.write('\n')
                        # tsv_writer.writerow(js['text'].encode('utf-8'))


if __name__ == '__main__':
    """
    usage: python make_tweets.py [ORIG_RECORD_FILE] [TWEETS_META_CUSTOM_LEN_FILE] [TWEETS_TEXT_CUSTOM_LEN_FILE] [LINES_TO_PROCESS]
    1. Iterate over original record file (ORIG_RECORD_FILE)
    2. Print to file (TWEETS_META_CUSTOM_LEN_FILE) custom lines (LINES_TO_PROCESS) of tweets meta (for debug purposes) for the next step
    3. Iterate over (TWEETS_META_CUSTOM_LEN_FILE) and print only the text to (TWEETS_TEXT_CUSTOM_LEN_FILE)
    """
    orig_file, tweets_meta_file, tweets_text_file, lines_to_process = getArgs()
    extractCustomLengthTweets(orig_file, tweets_meta_file, lines_to_process)
    extractTextFromTweets(tweets_meta_file, tweets_text_file)