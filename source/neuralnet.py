import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

class DRAW(object):

    def __init__(self, height, width, batch_size=100, sequence_length=10, learning_rate=1e-3, read_atten=False):

        self.height, self.width = height, width
        self.batch_size, self.sequence_length, self.learning_rate = batch_size, sequence_length, learning_rate

        self.attention_n = 5
        self.n_hidden = 256
        self.n_z = 10
        self.share_parameters = False

        self.canvas_seq = [0] * self.sequence_length # sequence of canvases
        self.mu, self.sigma = [0] * self.sequence_length, [0] * self.sequence_length

        self.inputs = tf.placeholder(tf.float32, [None, self.height*self.width]) # input (batch_size * img_size)
        self.z_noise = tf.random_normal((self.batch_size, self.n_z), mean=0, stddev=1) # Qsampler noise

        self.lstm_enc = tf.nn.rnn_cell.LSTMCell(num_units=self.n_hidden, state_is_tuple=True) # encoder Op
        self.lstm_dec = tf.nn.rnn_cell.LSTMCell(num_units=self.n_hidden, state_is_tuple=True) # decoder Op

        h_dec_prev = tf.zeros((self.batch_size, self.n_hidden))
        enc_state = self.lstm_enc.zero_state(self.batch_size, tf.float32)
        dec_state = self.lstm_dec.zero_state(self.batch_size, tf.float32)

        x = self.inputs
        for t in range(self.sequence_length):
            # error image + original image
            if(t==0): c_prev = tf.zeros((self.batch_size, self.height*self.width))
            else: self.canvas_seq[t-1]
            x_hat = x - tf.sigmoid(c_prev)

            if(read_atten): conseq = self.attention_read(x, x_hat, h_dec_prev) # read the image with attention
            else: conseq = self.concat_sequence(x, x_hat) # read the image

            self.mu[t], self.sigma[t], enc_state = self.encode(enc_state, tf.concat([conseq, h_dec_prev], 1))
            z = self.sample_latent(self.mu[t], self.sigma[t])
            h_dec, dec_state = self.decode(dec_state, z)

            print("Sequence %d" %(t))
            print("Input > Concat > Latent > Output")
            print(x.shape, conseq.shape, z.shape, h_dec.shape)

            self.canvas_seq[t] = c_prev + self.reconstruction(h_dec)
            h_dec_prev = h_dec
            self.share_parameters = True # from now on, share variables

        # calculate loss from final sequence
        self.final_sequence = tf.nn.sigmoid(self.canvas_seq[-1])
        # Reconstructor. E[log P(X|z)]
        self.loss_recon = tf.reduce_sum(self.inputs * tf.log(self.final_sequence) + (1 - self.inputs) * tf.log(1 - self.final_sequence), [1])

        # Regularizer. D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
        kl_list = [0]*self.sequence_length
        for t in range(self.sequence_length):
            tmp_mu = tf.square(self.mu[t])
            tmp_sigma = tf.square(self.sigma[t])
            kl_list[t] = 0.5 * tf.reduce_sum(tf.square(tmp_mu) + tf.square(tmp_sigma) - tf.log(1e-8 + tf.square(tmp_sigma)) - 1, [1])
        self.loss_kl = tf.reduce_mean(tf.add_n(kl_list)) # element wise sum using tf.add_n

        self.loss_recon = -tf.reduce_mean(self.loss_recon)
        self.loss_kl = tf.reduce_mean(self.loss_kl)
        self.ELBO = -self.loss_recon - self.loss_kl
        self.loss_total = -self.ELBO

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_total)

        tf.summary.scalar('loss_recon', self.loss_recon)
        tf.summary.scalar('loss_kl', self.loss_kl)
        tf.summary.scalar('loss_total', self.loss_total)
        self.summaries = tf.summary.merge_all()

    def sample_latent(self, mu, sigma): return mu + sigma * self.z_noise

    def concat_sequence(self, inputs, x_hat): return tf.concat([inputs, x_hat], 1)

    def reconstruction(self, hidden_layer):

        with tf.variable_scope("write", reuse=self.share_parameters):
            decoded_image_portion = self.fully_connected(hidden_layer, self.n_hidden, self.height*self.width)

        return decoded_image_portion

    def xavier_init(self, size):

        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)

        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def fully_connected(self, inputs, in_dim, out_dim, scope=None, with_w=False):

        with tf.variable_scope(scope or "Linear"):
            W = tf.Variable(self.xavier_init([in_dim, out_dim]))
            b = tf.Variable(tf.zeros(shape=[out_dim]))

            if with_w: return tf.matmul(inputs, W) + b, W, b
            else: return tf.matmul(inputs, W) + b

    def encode(self, prev_state, inputs):

        with tf.variable_scope("encoder", reuse=self.share_parameters):
            hidden_layer, next_state = self.lstm_enc(inputs, prev_state)

        with tf.variable_scope("mu", reuse=self.share_parameters):
            mu = self.fully_connected(hidden_layer, self.n_hidden, self.n_z)
        with tf.variable_scope("sigma", reuse=self.share_parameters):
            sigma = self.fully_connected(hidden_layer, self.n_hidden, self.n_z)

        return mu, sigma, next_state

    def decode(self, prev_state, latents):

        with tf.variable_scope("decoder", reuse=self.share_parameters):
            hidden_layer, next_state = self.lstm_dec(latents, prev_state)

        return hidden_layer, next_state

    def filter_img(self, inputs, Fx, Fy, gamma): # apply parameters for patch of gaussian filters

        Fxt = tf.transpose(Fx, perm=[0,2,1])
        inputs = tf.reshape(inputs, [-1, self.height, self.width])

        glimpse = tf.batch_matmul(Fy, tf.batch_matmul(img, Fxt)) # gaussian patches
        glimpse = tf.reshape(glimpse, [-1, self.attention_n**2])

        return glimpse * tf.reshape(gamma, [-1, 1]) # rescale

    def attention_read(self, inputs, x_hat, h_dec_prev):

        Fx, Fy, gamma = self.attn_window("read", h_dec_prev)

        x = self.filter_img(inputs, Fx, Fy, gamma)
        x_hat = self.filter_img(x_hat, Fx, Fy, gamma)

        return self.concat_sequence(x, x_hat)

    def attention_write(self, hidden_layer):

        with tf.variable_scope("writeW", reuse=self.share_parameters):
            w = self.fully_connected(hidden_layer, self.n_hidden, self.attention_n**2)
        w = tf.reshape(w, [self.batch_size, self.attention_n, self.attention_n])
        Fx, Fy, gamma = self.attn_window("write", hidden_layer)
        Fyt = tf.transpose(Fy, perm=[0,2,1])
        wr = tf.batch_matmul(Fyt, tf.batch_matmul(w, Fx))
        wr = tf.reshape(wr, [self.batch_size, self.height*self.width])

        return wr * tf.reshape(1.0/gamma, [-1, 1])

    def attn_window(self, scope, h_dec):
        with tf.variable_scope(scope, reuse=self.share_parameters):
            parameters = dense(h_dec, self.n_hidden, 5)
        # center of 2d gaussian on a scale of -1 to 1
        gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(1,5,parameters)

        # move gx/gy to be a scale of -imgsize to +imgsize
        gx = (self.width+1)/2 * (gx_ + 1)
        gy = (self.height+1)/2 * (gy_ + 1)

        sigma2 = tf.exp(log_sigma2)
        # distance between patches
        delta_x = (self.width - 1) / ((self.attention_n-1) * tf.exp(log_delta))
        delta_y = (self.width - 1) / ((self.attention_n-1) * tf.exp(log_delta))
        # returns [Fx, Fy, gamma]
        return self.filterbank(gx, gy, sigma2, delta_x, delta_y) + tf.exp(log_gamma)

    # Construct patches of gaussian filters
    def filterbank(self, gx, gy, sigma2, delta_x, delta_y):
        # 1 x N, look like [[0,1,2,3,4]]
        grid_i = tf.reshape(tf.cast(tf.range(self.attention_n), tf.float32),[1, -1])
        # individual patches centers
        mu_x = gx + (grid_i - self.attention_n/2 - 0.5) * delta_x
        mu_y = gy + (grid_i - self.attention_n/2 - 0.5) * delta_y
        mu_x = tf.reshape(mu_x, [-1, self.attention_n, 1])
        mu_y = tf.reshape(mu_y, [-1, self.attention_n, 1])
        # 1 x 1 x imgsize, looks like [[[0,1,2,3,4,...,27]]]
        im = tf.reshape(tf.cast(tf.range(self.img_size), tf.float32), [1, 1, -1])
        # list of gaussian curves for x and y
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])
        Fx = tf.exp(-tf.square((im - mu_x) / (2*sigma2)))
        Fy = tf.exp(-tf.square((im - mu_x) / (2*sigma2)))
        # normalize area-under-curve
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),1e-8)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),1e-8)
        return Fx, Fy
