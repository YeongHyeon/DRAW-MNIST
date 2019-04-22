import argparse

import tensorflow as tf

import source.neuralnet_resnet34 as nn
import source.datamanager as dman
import source.tf_process as tfp

def main():

    dataset = dman.DataSet()

    neuralnet = nn.DRAW()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # tfp.training(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch, dropout=FLAGS.dropout)
    # tfp.validation(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='Number of epoch for training')
    parser.add_argument('--batch', type=int, default=200, help='Mini-batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')

    FLAGS, unparsed = parser.parse_known_args()
