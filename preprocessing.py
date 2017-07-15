import argparse
import os

from multiprocessing import Process, Queue
import re
import glob

import pickle
import numpy as np
import PIL
from PIL import Image
import math

def is_valid_path(arg):
    if not os.path.exists(arg):
        parser.error("The directory %s does not exist!" % arg)
    else:
        return arg

class FullPaths(argparse.Action):
    """Expand user- and relative-paths"""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.normpath(os.path.abspath(os.path.expanduser(values))))

def is_valid_type(arg):
    if not arg in ['.jpg', '.png']:
        parser.error("%s is not a valid extention!" % arg)
    else:
        return arg

def is_valid_format(arg):
    if not arg.upper() in ['PICKLE', 'TXT', 'H5PY'] or arg == None:
        parser.error("%s is not a valid extention!" % arg)
    else:
        return arg

parser = argparse.ArgumentParser(description="Preprocess datasets for machine learning.")
parser.add_argument("-i", dest="path", required=False, help="specify path to database directory", metavar="PATH", action=FullPaths, type=is_valid_path, default='.')
parser.add_argument("-t", dest="type", required=False, help="specify data type", metavar="TYPE", type=is_valid_type, default='.jpg')
parser.add_argument("-s", dest="size", required=False, help="specify desired data size in pixels", metavar="SIZE", type=int, default='224')
parser.add_argument("--g", dest="greyscale", required=False, help="specify if image is grayscale", default=False, action='store_true')
parser.add_argument("-save", dest="save", required=False, help="choose format for saving processed images", type=is_valid_format, default=None)
parser.add_argument("--shuffle", dest="shuffle", required=False, help="specify whether to shuffle images in pickle files", default=False, action='store_true')
parser.add_argument("--v", dest="verify", required=False, help="verify that images have been saved properly", default=False, action='store_true')

args = parser.parse_args()

PATH = args.path

PATTERN = '*' + args.type
SIZE = args.size
NUM_CHANNELS = 1 if args.greyscale else 3
SAVE = args.save
SHUFFLE = args.shuffle
VERIFY = args.verify

def natural_key(string_):
    """
    Define sort key that is integer-aware
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def norm_image(img):
    """
    Normalize PIL image

    Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
    """
    img_y, img_b, img_r = img.convert('YCbCr').split()

    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    if NUM_CHANNELS == 1:
        img_nrm = img_ybr.convert('L')

    else:
        img_nrm = img_ybr.convert('RGB')

    return img_nrm


def resize_image(img, size):
    """
    Resize PIL image

    Resizes image to be square with sidelength size. Pads with black if needed.
    """
    # Resize
    n_x, n_y = img.size
    if n_y > n_x:
        n_y_new = size
        n_x_new = int(size * n_x / n_y + 0.5)
    else:
        n_x_new = size
        n_y_new = int(size * n_y / n_x + 0.5)

    img_res = img.resize((n_x_new, n_y_new), resample=PIL.Image.BICUBIC)

    # Pad the borders to create a square image
    if NUM_CHANNELS == 1:
        img_pad = Image.new('L', (size, size), 128)
    else:
        img_pad = Image.new('RGB', (size, size), (128, 128, 128))

    ulc = ((size - n_x_new) // 2, (size - n_y_new) // 2)
    img_pad.paste(img_res, ulc)

    return img_pad


def prep_images(images, out_dir):
    """
    Preprocess images

    Reads images in paths, and writes to out_dir

    """

    data = np.ndarray(shape=(len(images), SIZE, SIZE, NUM_CHANNELS), dtype=np.uint8)

    for count, path in enumerate(images):
        if count % 100 == 0:
            print(path)
        img = Image.open(path)
        img_nrm = norm_image(img)
        img_res = resize_image(img_nrm, SIZE)
        basename = os.path.basename(path)
        path_out = os.path.join(out_dir, basename)
        img_res.save(path_out)
        data[count, :, :, :] = np.reshape(np.asarray(img_res), (SIZE, SIZE, NUM_CHANNELS))

    return data

def main():

    """Main program for running from command line"""

    # Get the paths to all the image files

    images = sorted(glob.glob(os.path.join(PATH, PATTERN)), key=natural_key)

    # Make the output directories
    dir_name = os.path.basename(PATH) + "_" + ("G" if NUM_CHANNELS == 1 else "RGB") + '{}'.format(SIZE)
    out_dir = os.path.join(PATH, '..', dir_name)

    os.makedirs(out_dir, exist_ok=True)

    # Preprocess the training files

    data = prep_images(images, out_dir)

    return data, out_dir

if __name__ == '__main__':
    dataset, dir = main()

    if SAVE is not None and SHUFFLE:
        np.random.shuffle(dataset)

    if SAVE.upper() == "PICKLE":
        size = dataset.nbytes

        n = math.ceil(size / 4E9)

        for i in range(n):
            save = dataset[i * len(dataset) // n : (i + 1) * len(dataset) // n]

            name = os.path.basename(dir) + "_{}" + ".pickle"

            with open(os.path.join(dir, name.format(i)), 'wb') as f:
                pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)

            del save # for memory storage

        print("Pickling complete!")

        with open(os.path.join(dir, 'pickleinfo.txt'), 'w') as f:
            f.write("Number of files: {}\nTotal array size: {}\nArray shape: {}\n".format(n, size, dataset.shape))
            for i in range(n):
                f.write(os.path.join(dir, name.format(i)) + '\n')

    if SAVE.upper() == "TXT":

        name = os.path.basename(dir) + ".txt"

        with open(os.path.join(dir, name), 'wb') as f:
            f.write(str.encode("# Array shape: {0}\n".format(dataset.shape)))
            for img in dataset:
                f.write(str.encode("\n# New image\n"))
                for i in range(NUM_CHANNELS):
                    f.write(str.encode("\n# New slice\n\n"))
                    np.savetxt(f, img[:,:,i], fmt='%3i')

        print("Saving complete!")

    if SAVE.upper() == "H5PY":

        import h5py

        name = os.path.basename(dir) + ".h5"

        #labels = np.zeros(shape=(len(dataset),))
        #labels[len(dataset) // 2:] = 1

        with h5py.File(os.path.join(dir, name), 'w') as hf:
            hf.create_dataset('test_dataset',  data=dataset)
            #hf.create_dataset('train_labels', data=labels)

    if VERIFY:

        if SAVE.upper() == "TXT":

            print("Verifying file!")

            from ast import literal_eval as make_tuple

            name = os.path.basename(dir) + ".txt"
            path = os.path.join(dir, name)
            data = np.loadtxt(path)
            with open(path, 'r') as f:
                shape = make_tuple(f.readline()[15:])

            print(shape)

            data = data.reshape(shape)

            assert np.all(data == dataset)

            print("Verified!")

        if SAVE.upper() == "PICKLE":

            print("Verifying file!")

            from ast import literal_eval as make_tuple

            with open(os.path.join(dir, 'pickleinfo.txt'), 'r') as f:
                n = int(f.readline()[17:])
                size = int(f.readline()[18:])
                shape = make_tuple(f.readline()[13:])

            data_list = []

            for i in range(n):
                with open(os.path.join(dir, name.format(i)), 'rb') as f:
                    data = pickle.load(f)
                    data_list.append(data)

            dataset = np.vstack(data_list)

            data = data.reshape(shape)

            assert np.all(data == dataset)
            assert size == data.nbytes
            assert shape == data.shape

            print("Verified!")

        if SAVE.upper() == "H5PY":

            name = os.path.basename(dir) + ".h5"

            with h5py.File(os.path.join(dir, name), 'r') as f:
                data = f[os.path.basename(dir)][:]

                assert(np.all(data == dataset))

                print("Verified!")


