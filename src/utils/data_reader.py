
import random
from os import listdir
from os.path import isfile, join

import numpy as np
import tensorflow as tf


def gen_data(data_reader, label_reader, filename_queue):
    while True:

        _, value = data_reader.read(filename_queue)

        value = tf.image.decode_jpeg(value).eval()

        image = np.reshape(value, (64, 64, 1))
        label = next(label_reader)
        yield image, label

def gen_data_batch(data_reader, label_reader, train_filename_queue, batch_size=32):
    data_gen = gen_data(data_reader, label_reader, train_filename_queue)
    while True:
        image_batch = []
        label_batch = []
        for _ in range(batch_size):

            try:
                image, label = next(data_gen)

            except StopIteration as e:
                break


            image_batch.append(image)
            label_batch.append(label)
        yield np.array(image_batch), np.array(label_batch)


def get_readers(main_dir):

    diretories = listdir(main_dir)

    filenames = []
    labels = []

    class_index = 0
    for dir in diretories:

        current_dir = main_dir + "/" + dir

        class_index = int(dir.split('_')[0][-1:]);

        current_filenames = [current_dir + "/" + f for f in listdir(current_dir) if isfile(join(current_dir, f))]
        current_labels = [class_index]*len(current_filenames)

        filenames.extend(current_filenames)
        labels.extend(current_labels)


    combined = list(zip(filenames, labels))
    random.shuffle(combined)

    filenames[:], labels[:] = zip(*combined)


    one_hot_labels = []

    for label in labels:

        one_hot_label = [0]*10
        one_hot_label[label] = 1

        one_hot_labels.append(one_hot_label)


    data_reader = tf.WholeFileReader()

    label_reader = (l for l in one_hot_labels)

    filename_queue = tf.train.string_input_producer(filenames)

    return data_reader, label_reader, filename_queue, len(filenames)

