import matplotlib
import glob
import os
import tensorflow as tf
import numpy as np
import time
import matplotlib.image as mpimg


with tf.device('/cpu:0'):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # 读取图片
    def read_img(path):
        cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
        imgs = []
        labels = []
        for idx, folder in enumerate(cate):
            for im in glob.glob(folder + '/*.csv'):
                # print('reading the images:%s' % (im))
                # img = mpimg.imread(im)
                mat = np.loadtxt(open(im, "rb"), delimiter=",", skiprows=0)
                # img = transform.resize(img, (w, h))
                imgs.append(mat)
                labels.append(idx)
        return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

    def read_img2(path):
        mats = []
        files = os.listdir(path)  # 采用listdir来读取所有文件
        # files.sort(key=lambda x:int(x[:-4]))
        for file_ in files:
            if not os.path.isdir(path + file_):
                mat = np.loadtxt(open(path + file_, "rb"), delimiter=",", skiprows=0)
                # print('reading the images:%s' % (path+file_))
                mats.append(mat)

        return np.asarray(mats, np.float32)

    # 定义一个函数，按批次取数据
    def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]


    def mlp(_x):

        # 全连接层
        dense31 = tf.layers.dense(inputs=_x, units=100, activation=tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        drop33 = tf.layers.dropout(dense31, 0.2)
        logits = tf.layers.dense(inputs=drop33, units=2, activation=None,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        _y = tf.nn.softmax(logits)

        return _y, logits


    # --------------------------- 生成训练测试数据 -----------------------------------
    # path = 'd:\\Program Files/MATLAB/NeuroTIS-CodingWindowChoice/TrainBG96_84/'
    # test_path = 'd:\\Program Files/MATLAB/NeuroTIS-CodingWindowChoice/TestBG96_84/'
    path = 'd:\\tis/Content-CNN/train1_22C/'
    # test_path = 'd:\\Program Files/MATLAB/NeuroTIS/TestBG96_84/'
    test_path = 'd:\\tis/Content-CNN/test1_22C/'

    # 将所有的图片resize成100*100
    w = 4
    h = 91
    c = 1
    pw = 1.0
    n_epoch = 10
    batch_size = 500


    train_dir0 = path + '/0/'
    train_dir1 = path + '/1/'

    test_dir0 = test_path + '/0/'
    test_dir1 = test_path + '/1/'


    train_l0 = 882899
    # train_l0 = len([name for name in os.listdir(train_dir0) if os.path.isfile(os.path.join(train_dir0, name))])

    train_l1 = 882911
    # train_l1 = len([name for name in os.listdir(train_dir1) if os.path.isfile(os.path.join(train_dir1, name))])
    test_l0 = 313807
    # test_l0 =  len([name for name in os.listdir(test_dir0) if os.path.isfile(os.path.join(test_dir0, name))])
    test_l1 = 313807
    # test_l1 = len([name for name in os.listdir(test_dir1) if os.path.isfile(os.path.join(test_dir1, name))])

    data, label = read_img(path)
    data = np.reshape(data,[train_l0+train_l1,h,w,1])
    test_data, test_label = read_img(test_path)
    test_data = np.reshape(test_data, [test_l0+test_l1, h, w, 1])



    # ------------------------------- 构建网络 -------------------------------------
    # 占位符
    x = tf.placeholder(tf.float32, shape=[None, h, w, c], name='x')
    # x2 = tf.placeholder(tf.float32,shape=[None, h2],name = 'x2')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

    x_ = tf.reshape(x,[-1,91*4])

    y_hat, logits = mlp(x_)

    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.one_hot(y_,2),logits=logits,pos_weight=pw))

    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # auc = tf.metrics.auc(y_[:,1],logits[:,1])

    # 训练和测试数据，可将n_epoch设置更大一些
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        start_time = time.time()
        # training
        train_loss, train_acc, train_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(data, label, batch_size, shuffle=True):
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            train_batch += 1
        # validation
        val_loss, val_acc, val_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(test_data, test_label, batch_size, shuffle=False):
            err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err
            val_acc += ac
            val_batch += 1
        print("(%d/%d) train loss: %f, train acc: %f, validation loss: %f ,validation acc: %f"
              % (n_epoch, epoch + 1, train_loss / train_batch, train_acc / train_batch, val_loss / val_batch,
                 val_acc / val_batch))

    pred = np.zeros((test_l1+test_l0,2))
    i = 0
    for x_val_a, y_val_a in minibatches(test_data, test_label, 1, shuffle=False):
        pred[i,0:2] = sess.run(y_hat, feed_dict={x: x_val_a, y_: y_val_a})
        i = i + 1
    print(pred)
    np.savetxt("pred1.csv", pred, delimiter=",")
    np.savetxt("test_label1.csv", test_label, delimiter=",")

    sess.close()
