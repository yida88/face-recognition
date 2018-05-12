import tensorflow as tf
import numpy as np
from alenexcnn import CnnChaoCan,faceCNN
import loaddata
def train():
    best_acc=0.0
    saver=tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        writer=tf.summary.FileWriter('logs',sess.graph)
        sess.run(init)
        c = []
        X_train,y_train=loaddata.load_data()
        total_batch = int(X_train.shape[0] / config.batch_size)
        for i in range(config.training_iters):
            avg_cost = 0
            for batch in range(total_batch):
                batch_x = X_train[batch * config.batch_size: (batch + 1) * config.batch_size, :]
                batch_y = y_train[batch *config.batch_size: (batch + 1) * config.batch_size, :]
                _, co = sess.run([model.optimizer, model.cost], feed_dict={model.x:  batch_x, model.y: batch_y,model.keep_prob:0.5})

                avg_cost += co

                accuet, out = sess.run([model.accuracy, model.softmax], feed_dict={model.x: batch_x, model.y: batch_y,model.keep_prob:1.0})
                print( "train accuracy=" + "{:.6f}".format(accuet))
            #print(out)
            c.append(avg_cost)
            if (i + 1) % config.display_step == 0:
                print("Iter " + str(i + 1) + ", Training Loss= " + "{:.6f}".format(avg_cost))
            # if i>13:
            #     if accuet>best_acc:
            #         best_acc=accuet
        saver.save(sess=sess, save_path="./ckpt/test-model.ckpt")

        for variable in tf.trainable_variables():
            print(variable)


        print("Optimization Finished!")
        writer.close()
def test():
    print("loading test data")
    test_batch=20
    x_test=loaddata.load_testdata()
    total_batch = int(x_test.shape[0] / test_batch)
    outtest=model.softmax
    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,"./ckpt/test-model.ckpt")
        for batch in range(total_batch):
            batch_x = x_test[batch * test_batch: (batch + 1) * test_batch, :]
            out=sess.run(outtest,feed_dict={model.x:batch_x,model.keep_prob:1.0})
            c=np.argmax(out,axis=1)
            print(out)
            print(out[6])
            print(out[6][66])
            print(c)


if __name__=='__main__':
    config=CnnChaoCan()
    model=faceCNN(config)
    #train()
    test()