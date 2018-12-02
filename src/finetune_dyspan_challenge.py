
import tensorflow as tf

import src.utils.data_reader as data_reader
import src.utils.network_model as network_model

train_data_reader, train_label_reader, train_filename_queue, training_instances_total = data_reader.get_readers("C:/Users/Marcelo d'Almeida/Downloads/dyspan_files-20181110T190747Z-001/dyspan_files/TCCN dataset/training")
test_data_reader, test_label_reader, test_filename_queue, testing_instance_total = data_reader.get_readers("C:/Users/Marcelo d'Almeida/Downloads/dyspan_files-20181110T190747Z-001/dyspan_files/TCCN dataset/testing")
#train_data_reader, train_label_reader, train_filename_queue, training_instances_total = data_reader.get_readers("../data/TCCN dataset/smoke_training")
#test_data_reader, test_label_reader, test_filename_queue, testing_instance_total = data_reader.get_readers("../data/TCCN dataset/smoke_testing")

batch_size = 32

images = tf.placeholder(tf.float32, [batch_size, 64, 64, 1])
labels = tf.placeholder(tf.float32, [batch_size, 10])

model = network_model.cnn_to_mlp(
    images,
    convs=[(48, 11, 4, True, True),
           (128, 5, 2, True, True),
           (192, 3, 1, False, False),
           (192, 3, 1, False, False),
           (128, 3, 1, True, False)],
    max_poolings=[(3, 2, 'valid'),
                  (3, 2, 'valid'),
                  (3, 2, 'valid')],
    layer_norm=[(5, 0.0001, 0.75),
                (5, 0.0001, 0.75)],
    hiddens=[(1024, True),
             (4096, True),
             (10, False)]
)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=labels), 0)
opt = tf.train.RMSPropOptimizer(0.001)
train_op = opt.minimize(loss)

correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

def train():

    with tf.Session() as sess:
        # Load the data
        sess.run(tf.global_variables_initializer())
#        network_model.load('../data/dyspan_challenge_network_weights.npy', sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        data_gen = data_reader.gen_data_batch(train_data_reader, train_label_reader, train_filename_queue, batch_size)
        for i in range(training_instances_total):
            np_images, np_labels = next(data_gen)

            if (len(np_images) != batch_size):
                break

            feed = {images: np_images, labels: np_labels}

            np_loss, np_pred, _ = sess.run([loss, model, train_op], feed_dict=feed)
            if i % 10 == 0:
                print('Iteration: ', i, np_loss)

        coord.request_stop()
        coord.join(threads)

def test():

    with tf.Session() as sess:
        # Load the data
        sess.run(tf.global_variables_initializer())
#        network_model.load('../data/dyspan_challenge_network_weights.npy', sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        data_gen = data_reader.gen_data_batch(test_data_reader, test_label_reader, test_filename_queue, batch_size)
        for i in range(testing_instance_total):
            np_images, np_labels = next(data_gen)

            if (len(np_images) != batch_size):
                break

            feed = {images: np_images, labels: np_labels}

            accuracy_result = sess.run(accuracy, feed_dict=feed)
            print('Iteration: ', i, accuracy_result)

        coord.request_stop()
        coord.join(threads)


def predict():

    with tf.Session() as sess:
        # Load the data
        sess.run(tf.global_variables_initializer())
#        network_model.load('../data/dyspan_challenge_network_weights.npy', sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        data_gen = data_reader.gen_data_batch(test_data_reader, test_label_reader, test_filename_queue, batch_size)
        np_images, _ = next(data_gen)

        feed = {images: np_images}

        np_pred = sess.run(model, feed_dict=feed)
        print(np_pred)

        coord.request_stop()
        coord.join(threads)


print('training')
train()
print('testing')
test()
