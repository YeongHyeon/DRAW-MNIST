import matplotlib
matplotlib.use('Agg')
import os, inspect, math, scipy.misc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def make_dir(path):

    try: os.mkdir(path)
    except: pass

def save_image(savepath, image): scipy.misc.imsave(savepath, image)

def sigmoid(x): return 1 / (1 + np.exp(x))

def make_canvas(images, size):

    h, w = images.shape[1], images.shape[2]
    canvas = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        x = int(idx % size[1])
        y = int(idx / size[1])
        ys, ye, xs, xe = y*h, y*h+h, x*w, x*w+w
        canvas[ys:ye, xs:xe] = image

    return canvas

def save_result(canvas_seq, height, width, batch_size, epoch, savedir="recon"):

    canvas_seq = np.asarray(canvas_seq)

    for cs_iter in range(canvas_seq.shape[0]):
        tmp_sequence = np.reshape(sigmoid(canvas_seq[cs_iter]), [-1, height, width])
        canvas = make_canvas(tmp_sequence, [batch_size, batch_size])
        save_image(os.path.join("%s" %(savedir), "epoch%06d_seq%03d.png" %(epoch, cs_iter)), canvas)

def training(sess, neuralnet, saver, dataset, epochs, batch_size, sequence_length, print_step=1):

    print("\n* Training to %d epochs (%d of minibatch size)" %(epochs, batch_size**2))

    make_dir(path="recon_tr")
    make_dir(path="recon_te")

    train_writer = tf.summary.FileWriter(PACK_PATH+'/Checkpoint')
    # iterations = int(dataset.num_tr/batch_size**2)
    tr_batch = 5500
    iterations = int(dataset.num_tr/tr_batch)
    not_nan = True
    for epoch in range(epochs):
        if(epoch % print_step == 0):
            neuralnet.batch_size = batch_size**2
            x_tr, _ = dataset.next_train(batch_size**2)
            x_te, _ = dataset.next_test(batch_size**2)

            canvas_seq_tr, loss_recon_tr, loss_kl_tr = sess.run([neuralnet.canvas_seq, neuralnet.loss_recon, neuralnet.loss_kl], feed_dict={neuralnet.inputs:x_tr})
            canvas_seq_te, loss_recon_te, loss_kl_te = sess.run([neuralnet.canvas_seq, neuralnet.loss_recon, neuralnet.loss_kl], feed_dict={neuralnet.inputs:x_te})
            print("Epoch %06d (%d iteration)\nTR Loss - Recon: %.5f  KL: %.5f" % (epoch, iterations*epoch, loss_recon_tr, loss_kl_tr))
            print("TE Loss - Recon: %.5f  KL: %.5f" % (loss_recon_te, loss_kl_te))

            save_result(canvas_seq=canvas_seq_tr, height=dataset.height, width=dataset.width, batch_size=batch_size, epoch=epoch, savedir="recon_tr")
            save_result(canvas_seq=canvas_seq_te, height=dataset.height, width=dataset.width, batch_size=batch_size, epoch=epoch, savedir="recon_te")

        for iteration in range(iterations):
            # x_tr, _ = dataset.next_train(batch_size**2)
            neuralnet.batch_size = tr_batch
            x_tr, _ = dataset.next_train(tr_batch)

            summaries = sess.run(neuralnet.summaries, feed_dict={neuralnet.inputs:x_tr})
            train_writer.add_summary(summaries, iteration+(epoch*iterations))

            loss_recon_tr, loss_kl_tr, _ = sess.run([neuralnet.loss_recon, neuralnet.loss_kl, neuralnet.optimizer], feed_dict={neuralnet.inputs:x_tr})
            if(math.isnan(loss_recon_tr) or math.isnan(loss_kl_tr)):
                not_nan = False
                break

        if(not_nan): saver.save(sess, PACK_PATH+"/Checkpoint/model_checker")
        else:
            print("Training is terminated by Nan Loss")
            break

def validation(sess, neuralnet, saver, dataset, batch_size):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    make_dir(path="recon_te_final")

    iterations = int(dataset.num_tr/batch_size**2)
    loss_recon_tot, loss_kl_tot = [], []
    for iteration in range(iterations):
        x_te, _ = dataset.next_test(batch_size**2)
        canvas_seq_te, loss_recon_te, loss_kl_te = sess.run([neuralnet.canvas_seq, neuralnet.loss_recon, neuralnet.loss_kl], feed_dict={neuralnet.inputs:x_te})
        save_result(canvas_seq=canvas_seq_te, height=dataset.height, width=dataset.width, batch_size=batch_size, epoch=0, savedir="recon_te_final")
        loss_recon_tot.append(loss_recon_te)
        loss_kl_tot.append(loss_kl_te)

    loss_recon_tot = np.asarray(loss_recon_tot)
    loss_kl_tot = np.asarray(loss_kl_tot)
    print("Recon:%.5f  KL:%.5f" %(loss_recon_tot.mean(), loss_kl_tot.mean()))
