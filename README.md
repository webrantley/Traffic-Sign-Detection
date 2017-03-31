# Traffic-Sign-Detection

Models and testbed for the German Traffic Sign Recognition Benchmark (gtsrb). Users will have to download the dataset, format the training and test sets as Torch tensor files (.t7), and place them in a data directory.

#### Models

cifar - The baseline model for most convnets. A simple multi-layer net, with 16 features in the first layer, and 128 features in the second layer.

yann - A simplified version of Dr. LeCunn's approach to this benchmark [Link](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

idsia - An implementation of the winning submission of the original benchmark. [Link](http://people.idsia.ch/~juergen/nn2012traffic.pdf)

DNN - An attempt to improve on idsia by adding more features, intended to be used with a larger resizing of the original data.
