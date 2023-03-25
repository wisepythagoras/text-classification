import os
import sys
import re
import zipfile
import requests
import random
import shutil

def clean_str(s: str) -> str:
    s = s.lower()
    s = s.replace('…', '...')
    s = s.replace('\\u2019', "'")
    s = s.replace('\\u002c', ",")
    s = s.replace('&amp;', ' and ')
    s = s.replace('\t', ' ')
    s = s.replace('"', '')
    s = s.replace('\\', '')
    s = s.replace('/', '')
    s = re.sub(r'www\.[^\.]+\.[^\b]+', '', s)
    s = re.sub(r'\\?"\\?""""', '', s)
    s = re.sub(r'&(lt|gt);', '', s)
    s = re.sub(r'isn\'?t', 'is not', s)
    s = re.sub(r'don\'?t', 'do not', s)
    s = re.sub(r'won\'?t', 'will not', s)
    s = re.sub(r'wouldn\'?t', 'would not', s)
    s = re.sub(r'aren\'?t', 'are not', s)
    s = re.sub(r'i\'?m', 'i am', s)
    s = re.sub(r'i\'?ll', 'i will', s)
    s = re.sub(r'i\'?ve', 'i have', s)
    s = re.sub(r'let\'s', 'let us', s)
    s = re.sub(r'(s?he)\'d', '\\1 would', s)
    s = re.sub(r'(there|who|it)\'s', '\\1 is', s)
    s = re.sub(r'(you|they|we)\'?re', '\\1 are', s)
    s = re.sub(r'https?://t.co/[^\s]+', '', s)
    s = re.sub(r'@[^\s]+', '', s)

    s = re.sub(r'([a-z0-9])([:.?!\"/,)])', '\\1 \\2', s)
    s = re.sub(r'([:.?!\"/,(])([a-z0-9])', '\\1 \\2', s)

    return s.strip()

def read_dataset_file(path: str, name: str) -> list[list[str]]:
    labels_f = open('{}/{}_labels.txt'.format(path, name), 'r')
    labels = labels_f.readlines()

    texts_f = open('{}/{}_text.txt'.format(path, name), 'r')
    texts = texts_f.readlines()

    return [[clean_str(texts[i]), labels[i].strip()] for i in range(len(labels))]


if __name__ == '__main__':
    no_redownload = False

    try:
        no_redownload = sys.argv.index('--no-redownload') >= 0
    except:
        pass

    file_url = 'https://github.com/cardiffnlp/tweeteval/archive/refs/heads/main.zip'
    data_path = './data'
    file_path = '{}/tweeteval.zip'.format(data_path)
    extracted_path = '{}/tweeteval-main'.format(data_path)

    if not no_redownload:
        if os.path.exists(extracted_path):
            shutil.rmtree(extracted_path)
            print('Cleaned up previous version')

        zipFile = requests.get(file_url)
        open(file_path, 'wb').write(zipFile.content)
        print('Downloaded the file')

        with zipfile.ZipFile(file_path, 'r') as zipFile:
            zipFile.extractall(data_path)
            print('The file was extracted')

    train = read_dataset_file('{}/datasets/sentiment'.format(extracted_path), 'train')
    test = read_dataset_file('{}/datasets/sentiment'.format(extracted_path), 'test')
    val = read_dataset_file('{}/datasets/sentiment'.format(extracted_path), 'val')

    dataset = train + test + val

    print('\n'.join(['\t'.join(x) for x in dataset]))
