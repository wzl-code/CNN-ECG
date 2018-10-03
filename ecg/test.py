import tensorflow as tf
import  numpy as np
import PIL.Image as Image
from skimage import transform,data
tf.logging.set_verbosity(tf.logging.INFO)

def recognize(jpg_path, pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read()) #rb
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            input_x = sess.graph.get_tensor_by_name("input:0")
            print (input_x)
#            out_softmax = sess.graph.get_tensor_by_name("softmax:0")
#            print (out_softmax)
            out_label = sess.graph.get_tensor_by_name("output:0")
            print (out_label)

            img = Image.open(jpg_path)
            img = np.array(img)
            img=transform.resize(img,(128,128,1))
            #img = img.resize((28, 28,1),Image.ANTIALIAS)
            #plt.imshow(img)
            #plt.show()
            #img = img * (1.0 /255)
            img_out_softmax = sess.run(out_label, feed_dict={input_x:np.reshape(img, [-1,128,128,1])})

            print ("img_out_softmax:",img_out_softmax)
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            print ("prediction_labels:",prediction_labels)

recognize("D:/workplace/ecg/test_data/N/79-87_09.png", "D:/program/model/mnist_fang.pb")