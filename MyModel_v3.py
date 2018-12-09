
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import sklearn as sk
import tensorflow as tf
import os, random, time
from plots import plot_learning_curves
from methods import cnn_AlexNet, cnn_LeNet5

ill_size = 10000  # size of ill data to be used
normal_size = 10000  # size of normal data to be used
disease = "Cardiomegaly"  #  name of target disease
metadata_filepath = "data/Data_Entry_2017.csv"  # metadata file path
image_path = "images"  # images directory
summaries_dir = "logs" # training and testing logs directory
save_dir = "models" # direcotry to save and reload model parameters
GPU = True  # whether GPU is avaliable or not
method="LeNet5"  #AlexNet, LeNet5"
# Hyperparameters
n_classes = 2  # number of classes, here only 2 for normal and ill cases
batch_size = 32  # batch size for batch training
learning_rate = 0.001
alpha = 0.5
training_epochs = 1  # repeat training and testing times
display_step = 100  # number of steps to print traing/testing accuracy result
dropout = 0.1  # To prevent overfitting
ratio = 0.01  # variable initalization ratio 


"""
Sectio below load metadata and generate lables
"""
metadata = pd.read_csv(metadata_filepath)
full_path = np.vectorize(lambda image_path, image_name: os.path.join(image_path, image_name))
metadata["Image Index"] = full_path(image_path, metadata["Image Index"])

metadata["class"] = "NORMAL"
metadata.loc[metadata["Finding Labels"].str.contains(disease), ["class"]] = "ILL"
metadata = metadata[["Image Index","class"]]
ill_data = metadata[metadata["class"]=='ILL'][:ill_size]            #range, take out if run for all samples
normal_data = metadata[metadata["class"]=="NORMAL"][:normal_size]   #range
metadata = ill_data.append(normal_data)
train_metadata, test_metadata = train_test_split(metadata, test_size=0.05, shuffle=True)#split for train,test
train_metadata, valid_metadata = train_test_split(train_metadata, test_size=0.1, shuffle=True)#split for train,valid
total_count = metadata.shape[0]
normal_count = metadata[metadata["class"] == "NORMAL"].count()[0]
ill_count = metadata[metadata["class"] == "ILL"].count()[0]

class_weight = tf.constant([normal_count/total_count, ill_count/total_count])


"""
Section below define the procedure of slicing total data into batches and then laod image data
"""
with tf.device('/device:GPU:0' if GPU else '/cpu:0'):

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

    train_data = tf.data.Dataset.from_tensor_slices(
        (train_metadata["Image Index"].values, 
         pd.get_dummies(train_metadata["class"]).values))

    # for a small batch size
    train_data = train_data.map(_parse_function, num_parallel_calls=4)
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(1)
    
    evaluation_data = tf.data.Dataset.from_tensor_slices(
        (valid_metadata["Image Index"].values, 
         pd.get_dummies(valid_metadata["class"]).values))
    evaluation_data = evaluation_data.map(_parse_function, num_parallel_calls=4)
    evaluation_data = evaluation_data.batch(batch_size)
    evaluation_data = evaluation_data.prefetch(1)    
    

    test_data = tf.data.Dataset.from_tensor_slices(
        (test_metadata["Image Index"].values, 
         pd.get_dummies(test_metadata["class"]).values))
    test_data = test_data.map(_parse_function, num_parallel_calls=4)
    test_data = test_data.batch(batch_size)
    test_data = test_data.prefetch(1)
    
    iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                               train_data.output_shapes)
    x, y = iterator.get_next()

    train_init = iterator.make_initializer(train_data) # Inicializador para train_data
    evaluate_init = iterator.make_initializer(evaluation_data) # Inicializador para valid_data
    test_init = iterator.make_initializer(test_data) # Inicializador para test_data
    
    # Visualize input x
    tf.summary.image("input", x, batch_size)

    

"""
Section below define CNN structure, train procedure and test procedure
"""
with tf.device('/device:GPU:0' if GPU else '/cpu:0'):
    if method=="AlexNet":
        pred, keep_prob=cnn_AlexNet(x, ratio, n_classes)
        
    if method=="LeNet5":
        pred, keep_prob=cnn_LeNet5(x, ratio, n_classes)
    
    with tf.name_scope("cross_entropy"):
        # softmax
        softmax = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
        cost = tf.reduce_mean(softmax)
        tf.summary.scalar("cross_entropy", cost)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

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
with tf.device('/device:GPU:0' if GPU else '/cpu:0'):
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
    train_costs, train_accuracies=[],[]
    valid_costs, valid_accuracies=[],[]

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
    for i in range(training_epochs):
        print("epoch: {}\n".format(i))   
        print("\n")
        epoch_start = time.time()
        sess.run(train_init)
        epoch_step = 0
        avg_acc = 0
        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0
        avg_loss = 0
        try:
            while True:
                _, acc, y_pred, y_true, loss, summary_str = sess.run(
                    [optimizer, accuracy, predicted, actual, cost, summ],
                    feed_dict={keep_prob: 1-dropout},
                    options=run_options,
                    run_metadata = run_metadata) 
                precision = sk.metrics.precision_score(y_true, y_pred)
                recall = sk.metrics.recall_score(y_true, y_pred)
                f1 = sk.metrics.f1_score(y_true, y_pred)
                train_writer.add_summary(summary_str, step)
        
                if step % display_step == 0:
                    train_writer.add_run_metadata(run_metadata,"step {}".format(step))
                    print("step: {}".format(step))
                    print("accuracy: {}".format(acc))
                    print("precision: {}".format(precision))
                    print("recall: {}".format(recall))
                    print("f1_score: {}".format(f1))
                    print("loss: {}".format(loss))
                    print("\n")
                avg_acc += acc
                avg_precision += precision
                avg_recall += recall
                avg_f1 += f1
                avg_loss += loss
                step += 1
                epoch_step += 1
                train_costs.append(avg_loss/epoch_step)
                train_accuracies.append(avg_acc/epoch_step)
        except tf.errors.OutOfRangeError:

            epoch_time = time.time() - epoch_start
            total_epoch_time += epoch_time
            print("epoch finished in {} seconds".format(epoch_time))
            print("Average epoch accuracy is {:.2f}%".format((avg_acc / epoch_step) * 100))
            print("Average epoch precision is {:.2f}%".format((avg_precision / epoch_step) *100))
            print("Average epoch recall is {:.2f}%".format((avg_recall / epoch_step) *100))
            print("Average epoch f1-score is {:.2f}".format((avg_f1 / epoch_step)))
            print("Average epoch loss is {:.2f}".format(avg_loss / epoch_step))
        # save trained model parameters
        save_name = disease + "_model"
        save_path = os.path.join(save_dir, save_name)
        saver.save(sess, save_path)
        print("Model saved in path: %s" % save_path)

        #Evaluation
        print("Evaluation\n")
        sess.run(evaluate_init)
        avg_acc = 0
        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0
        avg_loss = 0
        test_step=0
        try:
            while True:
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
                test_writer.add_summary(summary_str, test_step)
                valid_costs.append(avg_loss/epoch_step)
                valid_accuracies.append(acc/epoch_step)   
        except tf.errors.OutOfRangeError:           
            print("Average test set accuracy over {} iterations is {:.2f}%".format(test_step,(avg_acc / test_step) * 100))
            print("Average epoch precision is {:.2f}%".format((avg_precision / test_step) *100))
            print("Average epoch recall is {:.2f}%".format((avg_recall / test_step) *100))
            print("Average epoch f1-score is {:.2f}".format((avg_f1 / test_step)))
            print("Average test set loss over {} iterations is {:.2f}".format(test_step,(avg_loss / test_step)))
            print("\n")
         
        #plot
        plot_learning_curves(train_costs, valid_costs, train_accuracies, valid_accuracies, loss_fig=disease+"_Loss.png", accuracy_fig=disease+"_accuracy.png")      
    
        # Test            
    
        print("Test\n")
        sess.run(test_init)
        avg_acc = 0
        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0
        avg_loss = 0
        test_step=0
        try:
            while True:
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
                test_writer.add_summary(summary_str, test_step) 
        except tf.errors.OutOfRangeError:           
            print("Average test set accuracy over {} iterations is {:.2f}%".format(test_step,(avg_acc / test_step) * 100))
            print("Average epoch precision is {:.2f}%".format((avg_precision / test_step) *100))
            print("Average epoch recall is {:.2f}%".format((avg_recall / test_step) *100))
            print("Average epoch f1-score is {:.2f}".format((avg_f1 / test_step)))
            print("Average test set loss over {} iterations is {:.2f}".format(test_step,(avg_loss / test_step)))
            print("\n")
    

    print("Average epoch time: {} seconds".format(total_epoch_time/training_epochs))
    train_writer.add_run_metadata(run_metadata,"mySess")
    train_writer.close()
    test_writer.close()
    sess.close()




