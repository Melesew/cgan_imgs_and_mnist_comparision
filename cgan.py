
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


# In[2]:


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mb_size = 64
Z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128


# In[3]:


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# In[4]:


""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


# In[5]:


def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


# In[6]:


""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


# In[7]:


def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


# In[8]:


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# In[9]:


def plot(samples):
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(16, 16)
    gs.update(wspace=0.0005, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


# In[10]:


G_sample = generator(Z, y)
D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)


# In[11]:


D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))


# In[12]:


D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


# In[13]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[14]:


if not os.path.exists('cgan_imgs/'):
    os.makedirs('cgan_imgs/')


# In[15]:


for k in range(0, 10):
    if not os.path.exists('cgan_imgs/{}/'.format(k)):
        os.makedirs('cgan_imgs/{}/'.format(k))
    
    i = 0
    
    # Generate images in every 100 iterations
    for it in range(250000):
        if it % 100 == 0:
            n_sample = 1

            Z_sample = sample_Z(n_sample, Z_dim)
            y_sample = np.zeros(shape=[n_sample, y_dim])
            y_sample[:, k] = 1


            samples = sess.run(G_sample, feed_dict={Z: Z_sample, y:y_sample})
            
            # Save the best 1500 images for each digit sothat we can use 1200 for training and 300 for testing later
            
            if (it >= 100000):
                fig = plot(samples)
                plt.savefig('cgan_imgs/{}/{}.png'.format(k, str(i).zfill(3)), bbox_inches='tight')
                i += 1
                plt.close(fig)                    

        X_mb, y_mb = mnist.train.next_batch(mb_size)

        Z_sample = sample_Z(mb_size, Z_dim)
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y:y_mb})

        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()   

