import argparse

import tensorflow as tf

import source.neuralnet as nn
import source.datamanager as dman
import source.tf_process as tfp

def main():

    dataset = dman.DataSet()
    neuralnet = nn.DRAW(height=dataset.height, width=dataset.width, batch_size=FLAGS.batch**2, sequence_length=FLAGS.seqlen, learning_rate=FLAGS.lr, attention=FLAGS.attention)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    tfp.training(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch, sequence_length=FLAGS.seqlen, print_step=10)
    tfp.validation(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, batch_size=FLAGS.batch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000, help='-')
    parser.add_argument('--batch', type=int, default=10, help='-')
    parser.add_argument('--seqlen', type=int, default=10, help='-')
    parser.add_argument('--lr', type=float, default=0.001, help='-')
    parser.add_argument('--attention', type=bool, default=False, help='-')

    FLAGS, unparsed = parser.parse_known_args()

    main()
