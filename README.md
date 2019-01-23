This is a TensorFlow implementation for "Multi-Granularity Hierarchical Attention Fusion Networks for Reading Comprehension and Question Answering" paper by Alibaba for Squad dataset. 

1) To train the network run the following command
python main.py --experiment_name=baseline --mode=train
2) For inference run the following command
python main.py --experiment_name=baseline --mode=show_examples
3) For running in Manual question and answer mode run the following command
python main.py --experiment_name=baseline --mode=compare


Note: this code is adapted in part from the Final Project (SQuAD) for [CS224n](http://web.stanford.edu/class/cs224n/), Winter 2018
