"""Adapted from a utility used in the Google Deep Learning Udacity
Course, available at https://www.udacity.com/course/deep-learning--ud730."""


import os
import sys
from six.moves.urllib.request import urlretrieve

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.'


def download_progress_hook(count, blocksize, totalSize):
    global last_percent_reported
    percent = int(count * blocksize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write('{}%'.format(percent))
            sys.stdout.flush()
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

        last_percent_reported = percent

def maybe_download(filename, expected_bytes, force=False):
    dest_filename = os.path.join(data_root, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
            'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename


#train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
