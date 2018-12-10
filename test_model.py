
import numpy as np 
import pandas as pd 
import sklearn as sk
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os, random, time

test_size = 1000 # number of test data points
disease = "Cardiomegaly"  #  name of target disease
metadata_filepath = "./data/Data_Entry_2017.csv"  # metadata file path
test_result_filepath = "test_result.csv" # file to save test result 
image_path = "./images"  # images directory
save_dir = "./models" # direcotry to save and reload model parameters
summaries_dir = "./logs" # training and testing logs directory
GPU = False  # whether GPU is avaliable or not
localization = True



# Hyperparameters
n_classes = 2  # number of classes, here only 2 for normal and ill cases
batch_size = 50  # batch size for batch training
learning_rate = 0.001
alpha = 0.5
epochs = 1  # repeat training and testing times
display_step = 10  # number of steps to print traing/testing accuracy result
dropout = 0.1  # To prevent overfitting
ratio = 0.01  # variable initalization ratio 


"""
Sectio below load metadata and generate lables
"""
metadata = pd.read_csv(metadata_filepath)
valid_images = pd.DataFrame(os.listdir(image_path), columns = ["Image Index"])
metadata = metadata.join(valid_images.set_index('Image Index'), on='Image Index', how = 'inner')
metadata = metadata[:test_size]
full_path = np.vectorize(lambda image_path, image_name: os.path.join(image_path, image_name))
metadata["Image Index"] = full_path(image_path, metadata["Image Index"])

metadata["class"] = "NORMAL"
metadata.loc[metadata["Finding Labels"].str.contains(disease), ["class"]] = "ILL"
metadata = metadata[["Image Index","class"]]
test_metadata = metadata

total_count = metadata.shape[0]
normal_count = metadata[metadata["class"] == "NORMAL"].count()[0]
ill_count = metadata[metadata["class"] == "ILL"].count()[0]

class_weight = tf.constant([normal_count/total_count, ill_count/total_count])





"""
Section below define the procedure of slicing total data into batches and then laod image data
"""
with tf.device('/cpu:0'):

    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels = 1)
        image_decoded = tf.image.resize_images(image_decoded,[256, 256])
        image_decoded = tf.cast(image_decoded, tf.float32)
        image_decoded = tf.image.per_image_standardization(image_decoded)
        #image_decoded.set_shape((256, 256, 1))
        return image_decoded, label


    test_data = tf.data.Dataset.from_tensor_slices(
        (test_metadata["Image Index"].values,
         pd.concat([pd.get_dummies(test_metadata["class"]).NORMAL,
                    pd.get_dummies(test_metadata["class"]).ILL],axis =1).values))
    test_data = test_data.map(_parse_function, num_parallel_calls=4)
    test_data = test_data.batch(batch_size)
    test_data = test_data.prefetch(1)
    
    iterator = tf.data.Iterator.from_structure(test_data.output_types, 
                                               test_data.output_shapes)
    x, y = iterator.get_next()

    test_init = iterator.make_initializer(test_data) # Inicializador para test_data
    
    # Visualize input x
    tf.summary.image("input", x, batch_size)

    

"""
Section below define CNN structure, train procedure and test procedure
"""
with tf.device('/device:GPU:0' if GPU else '/cpu:0'):
    def conv2d(img, w, b, k = 1):
        return tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, k, k, 1], padding='SAME'),b))

    def max_pool(img, k):
        return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    wc1 = tf.Variable(tf.random_normal([11, 11, 1, 32])*ratio, name="wc1")
    bc1 = tf.Variable(tf.random_normal([32])*ratio, name="bc1")
    # stride 64 x 64
    # pool 32 x 32
    wc2 = tf.Variable(tf.random_normal([3, 3, 32, 128])*ratio, name="wc2")
    bc2 = tf.Variable(tf.random_normal([128])*ratio, name="bc2")
    # pool 16 x 16
    wc3 = tf.Variable(tf.random_normal([3, 3, 128, 96])*ratio, name="wc3")
    bc3 = tf.Variable(tf.random_normal([96])*ratio, name="bc3")
    # pool 16 x 16
    wc4 = tf.Variable(tf.random_normal([3, 3, 96, 64])*ratio, name="wc4")
    bc4 = tf.Variable(tf.random_normal([64])*ratio, name="bc4")
    # pool 8x8
    
    wd1 = tf.Variable(tf.random_normal([8*8*64, 512])*ratio, name="wd1")
    bd1 = tf.Variable(tf.random_normal([512])*ratio, name="bd1")
    wd2 = tf.Variable(tf.random_normal([512, 256])*ratio, name="wd2")
    bd2 = tf.Variable(tf.random_normal([256])*ratio, name="bd2")
    wout = tf.Variable(tf.random_normal([256, n_classes])*ratio, name="wout")
    bout = tf.Variable(tf.random_normal([n_classes])*ratio, name="bout")
    
    # conv layer
    #x = tf.Print(x, [x])
    conv1 = conv2d(x,wc1,bc1, k = 4)
    conv1 = max_pool(conv1, k=2)
    # conv layer
    conv2 = conv2d(conv1,wc2,bc2)

    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 64*64 matrix.
    conv2 = max_pool(conv2, k=2)
    # conv2 = avg_pool(conv2, k=2)

    # dropout to reduce overfitting
    keep_prob = tf.placeholder(tf.float32)
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # conv layer
    conv3= conv2d(conv2,wc3,bc3)


    # dropout to reduce overfitting
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # conv layer
    conv4 = conv2d(conv3,wc4,bc4)

    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 64*64 matrix.
    conv4 = max_pool(conv4, k=2)

    # dropout to reduce overfitting
    conv4 = tf.nn.dropout(conv4, keep_prob)
    
    # conv4 shape [batch_size, 8, 8, 64]
    

    # fc 1
    dense1 = tf.reshape(conv4, [-1, wd1.get_shape().as_list()[0]])
    dense1 = tf.nn.tanh(tf.add(tf.matmul(dense1, wd1),bd1))
    dense1 = tf.nn.dropout(dense1, keep_prob)

    # fc 2
    dense2 = tf.reshape(dense1, [-1, wd2.get_shape().as_list()[0]])
    dense2 = tf.nn.tanh(tf.add(tf.matmul(dense2, wd2),bd2))
    dense2 = tf.nn.dropout(dense2, keep_prob)
   

    # prediction
    pred = tf.add(tf.matmul(dense2, wout), bout)



        
        

    #weighted_pred = tf.multiply(pred, class_weight)

    with tf.name_scope("cross_entropy"):
        # softmax
        softmax = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
        cost = tf.reduce_mean(softmax)
        tf.summary.scalar("cross_entropy", cost)


    with tf.name_scope("accuracy"):
        # Accuracy
        predicted = tf.argmax(pred, 1)
        actual = tf.argmax(y, 1)
        correct_pred = tf.equal(predicted, actual)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", accuracy)




"""
Section below initialize CNN glopbal variables and configuration
"""
with tf.device('/cpu:0'):
    # Get all summary
    summ = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement = True)
    saver = tf.train.Saver()
    save_name = disease + "_model"
    save_path = os.path.join(save_dir, save_name)




    
"""
Section below starts iteration of training and testing
"""
# Session start
with tf.Session(config=config) as sess:
    
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(summaries_dir + '/test', sess.graph)
    
    # Required to get the filename matching to run.
    try:
        saver.restore(sess, save_path)
        print("Model loaded with file path: %s" % save_path)
    except:
        sess.run(init)
        print("Saved model file path: %s doesn't exit, using random initialization" % save_path)

    total_epoch_time = 0
    step = 1
    # Compute epochs.
    for i in range(epochs):
        print("epoch: {}\n".format(i))
        print("\n")
 
        # Test            

        print("Test\n")
        sess.run(test_init)
        test_predict = []
        avg_acc = 0
        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0
        avg_loss = 0
        test_step=0
        try:
             while True:
                if (localization):
                    acc, y_pred, y_true, loss, summary_str = sess.run(
                        [accuracy, predicted, actual, cost, summ],
                        feed_dict={keep_prob: 1.})
                else :
                    acc, y_pred, y_true, loss, summary_str = sess.run(
                        [accuracy, predicted, actual, cost, summ],
                        feed_dict={keep_prob: 1.})
                precision = sk.metrics.precision_score(y_true, y_pred)
                recall = sk.metrics.recall_score(y_true, y_pred)
                f1 = sk.metrics.f1_score(y_true, y_pred)
                avg_acc += acc
                avg_precision += precision
                avg_recall += recall
                avg_f1 += f1
                avg_loss += loss
                test_step += 1
                test_predict += y_pred.tolist()
                test_writer.add_summary(summary_str, test_step)  
                print("accuracy: {}".format(acc))
                print("precision: {}".format(precision))
                print("recall: {}".format(recall))
                print("f1_score: {}".format(f1))
                print("loss: {}".format(loss))
                print("\n")
        except tf.errors.OutOfRangeError:
            print("Average test set accuracy over {} iterations is {:.2f}%".format(test_step,(avg_acc / test_step) * 100))
            print("Average epoch precision is {:.2f}%".format((avg_precision / test_step) *100))
            print("Average epoch recall is {:.2f}%".format((avg_recall / test_step) *100))
            print("Average epoch f1-score is {:.2f}".format((avg_f1 / test_step)))
            print("Average test set loss over {} iterations is {:.2f}".format(test_step,(avg_loss / test_step)))
            print("\n")
            
        if(i == epochs - 1):
            # save result result
            test_metadata["pred"] = test_predict
            


    print("Average epoch time: {} seconds".format(total_epoch_time/epochs))
    train_writer.add_run_metadata(run_metadata,"mySess")
    train_writer.close()
    test_writer.close()
    test_metadata.to_csv(disease+'_'+test_result_filepath)










