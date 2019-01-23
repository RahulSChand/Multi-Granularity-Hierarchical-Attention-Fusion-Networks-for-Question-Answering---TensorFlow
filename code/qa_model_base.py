

"""This file defines the top-level model"""

from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, SimpleSoftmaxLayerNew,BasicAttn,RNNEncoderLSTM

logging.basicConfig(level=logging.INFO)

from data_batcher import sentence_to_token_ids 

class PreQAModel(object):

    def __init__(self,emb_matrix,max_question_len,word2id):
        self.emb_matrix=emb_matrix
        self.max_question_len=max_question_len
        self.word2id=word2id
        self.new_qn_file_ids_tensor = tf.placeholder(tf.int32,shape=[None,self.max_question_len])
        self.manual_qn_file_ids_tensor = tf.placeholder(tf.int32,shape=[None,self.max_question_len])
        self.run_op = self.compare_questions_return()

    def compare_questions_preprocess(self,new_qn_file,manual_qn_file,manual_answer_file):

        new_qn = new_qn_file.readline()
        print(new_qn)
        #time.sleep(100)
        new_qn_file_tokens,new_qn_file_ids = sentence_to_token_ids(new_qn,self.word2id)


        new_qn_file_ids += [220] * (20 - len(new_qn_file_ids))
        manual_qn = manual_qn_file.readline()
        manual_qn_ids=[]
        while manual_qn:
            manual_qn_file_tokens,manual_qn_file_ids = sentence_to_token_ids(manual_qn,self.word2id)
            manual_qn_file_ids += [10] * (20 - len(manual_qn_file_ids))
            manual_qn_ids.append(manual_qn_file_ids)
            manual_qn=manual_qn_file.readline()
        new_qn_file_ids,manual_qn_ids = np.array(new_qn_file_ids),np.array(manual_qn_ids)
        new_qn_file_ids = np.expand_dims(new_qn_file_ids,axis=0)
        print(new_qn_file_ids.shape,manual_qn_ids.shape)
        return (new_qn_file_ids,manual_qn_ids)

    def compare_questions_return(self):

        embedding_matrix = tf.constant(self.emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)
        qn_new_emb_print = embedding_ops.embedding_lookup(embedding_matrix, self.new_qn_file_ids_tensor) # shape (batch_size, question_len, embedding_size)
        qn_man_emb_print = embedding_ops.embedding_lookup(embedding_matrix, self.manual_qn_file_ids_tensor) # shape (batch_size, question_len, embedding_size)

        qn_new_emb=tf.Print(qn_new_emb_print,[qn_new_emb_print])
        qn_man_emb=tf.Print(qn_man_emb_print,[qn_man_emb_print])
        print("**************************************")
        print(qn_new_emb.get_shape().as_list())
        print(qn_man_emb.get_shape().as_list())
        tile_tensor = tf.constant([8,1,1])
        qn_new_emb_tile = tf.tile(qn_new_emb,tile_tensor)

        dot_product_cal = tf.multiply(qn_man_emb,qn_new_emb_tile)
        reduce_sum = tf.reduce_sum(dot_product_cal,axis=2)
        reduce_sum = tf.reduce_sum(reduce_sum,axis=1)
        #reduce_sum_div = tf.reduce_sum(reduce_sum,axis=0)
        #reduce_sum_div=tf.expand_dims(reduce_sum_div,axis=0)
        #reduce_sum_div = tf.tile(reduce_sum_div,[8])
        #reduce_sum = tf.divide(reduce_sum,reduce_sum_div)
        #reduce_sum=tf.nn.softmax(reduce_sum)
        return reduce_sum

    def compare_questions(self,new_qn_file,manual_qn_file,manual_answer_file):

        new_qn_file_ids,manual_qn_ids=self.compare_questions_preprocess(new_qn_file,manual_qn_file,manual_answer_file)
        reduce_sum_output=0
        with tf.Session() as sess:
            input_feed={}
            input_feed[self.new_qn_file_ids_tensor]=new_qn_file_ids
            input_feed[self.manual_qn_file_ids_tensor]=manual_qn_ids
            reduce_sum_output = sess.run(self.run_op,input_feed)
        print(reduce_sum_output)

class QAModel(object):
    """Top-level Question Answering module"""

    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        """
        Initializes the QA model.

        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string)
          word2id: dictionary mapping word (string) to word idx (int)
          emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
        """
        print("Initializing the QAModel...")
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id

        # Add all parts of the graph
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()


    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())




    def add_embedding_layer(self, emb_matrix):
        """
        Adds word embedding layer to the graph.

        Inputs:
          emb_matrix: shape (400002, embedding_size).
            The GloVe vectors, plus vectors for PAD and UNK.
        """
        with vs.variable_scope("embeddings"):

            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)

            # Get the word embeddings for the context and question,
            # using the placeholders self.context_ids and self.qn_ids
            self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            self.qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size)


    def matching_function(self,Tilda,Normal,name,NormalShape):                  #For the Mathcing ('m' function in Fusion function)

        matrix_dimensions = tf.shape(Normal)
        batch_size,matrix_size,hidden_size = matrix_dimensions[0],matrix_dimensions[1],matrix_dimensions[2]


        Wf = tf.get_variable(name,shape=[NormalShape.get_shape().as_list()[1],4*NormalShape.get_shape().as_list()[1]],trainable=True)
        tf.summary.histogram("weightsGating"+name, Wf)


        #WfReshape = tf.transpose(Wf,perm=[1,0,2])

        nameBias = name+"bias"
        Bias = tf.get_variable(nameBias,shape=[1],trainable=True)
        tf.summary.histogram("Bias"+name, Bias)

        ElementWiseProduct = tf.multiply(Tilda,Normal)     #(batch,paragraph,hidden*2)

        #ElementWiseProduct = tf.reshape(ElementWiseProductTemp,[batch_size,matrix_size,hidden_size])

        SubtractMatrix = tf.subtract(Normal,Tilda)

        ConcatMatrix = tf.concat([tf.concat([tf.concat([Normal,Tilda],0),ElementWiseProduct],0),SubtractMatrix],0) #(batch,4*paragraph,hidden*2)

        
        print(Normal.get_shape().as_list())
        print(ConcatMatrix.get_shape().as_list())



        ConcatMatrixReshape = tf.reshape(tf.transpose(ConcatMatrix,perm=[1,0,2]),[4*matrix_size,batch_size*hidden_size])    #(4P,H*B)


        Matching1 = tf.matmul(Wf,ConcatMatrixReshape)   #(P,H*B)

        Matching2_new = tf.reshape(Matching1,[matrix_size,batch_size,hidden_size])  #(P,B,H)
        Matching2_transpose = tf.transpose(Matching2_new,perm=[1,0,2]) #(B,P,H)
        addBias = tf.add(Matching2_transpose,Bias)

        matchingOutput = tf.nn.tanh(addBias)

        return matchingOutput


    def gating_function(self,Tilda,Normal,name,NormalShape):                  #For the Mathcing ('m' function in Fusion function)
        matchingOutput = None
        with vs.variable_scope(name+"SCOPE"):

            matrix_dimensions = tf.shape(Normal)
            batch_size,matrix_size,hidden_size = matrix_dimensions[0],matrix_dimensions[1],matrix_dimensions[2]


            Wf = tf.get_variable(name,shape=[NormalShape.get_shape().as_list()[1],4*NormalShape.get_shape().as_list()[1]],trainable=True)
            tf.summary.histogram("weightsGating"+name, Wf)


            #WfReshape = tf.transpose(Wf,perm=[1,0,2])

            nameBias = name+"bias"
            Bias = tf.get_variable(nameBias,shape=[1],trainable=True)
            tf.summary.histogram("Bias"+name, Bias)
            ElementWiseProduct = tf.multiply(Tilda,Normal)     #(batch,paragraph,hidden*2)

            #ElementWiseProduct = tf.reshape(ElementWiseProductTemp,[batch_size,matrix_size,hidden_size])

            SubtractMatrix = tf.subtract(Normal,Tilda)

            ConcatMatrix = tf.concat([tf.concat([tf.concat([Normal,Tilda],0),ElementWiseProduct],0),SubtractMatrix],0) #(batch,4*paragraph,hidden*2)


            print(Normal.get_shape().as_list())
            print(ConcatMatrix.get_shape().as_list())




        ConcatMatrixReshape = tf.reshape(tf.transpose(ConcatMatrix,perm=[1,0,2]),[4*matrix_size,batch_size*hidden_size])    #(4P,H*B)


        Matching1 = tf.matmul(Wf,ConcatMatrixReshape)   #(P,H*B)

        Matching2_new = tf.reshape(Matching1,[matrix_size,batch_size,hidden_size])  #(P,B,H)
        Matching2_transpose = tf.transpose(Matching2_new,perm=[1,0,2]) #(B,P,H)
        addBias = tf.add(Matching2_transpose,Bias)

        matchingOutput = tf.nn.sigmoid(addBias)

        return matchingOutput

       



    def Fuse(self,Tilda,Normal,name1,name2,NormalShape):
        gatingOutput = self.gating_function(Tilda,Normal,name1,NormalShape)
        matchingOutput = self.matching_function(Tilda,Normal,name2,NormalShape)

        print(gatingOutput.get_shape().as_list())
        print(matchingOutput.get_shape().as_list())
        ##time.sleep(100)

        returnNewTildaPart1 = tf.multiply(gatingOutput,matchingOutput)
        returnNewTildaPart2 = tf.multiply(gatingOutput,tf.subtract(1.00,matchingOutput))

        returnNewTilda = tf.add(returnNewTildaPart1,returnNewTildaPart2)


        return returnNewTilda


        '''
    def gating_function(self,Tilda,Normal,name,NormalShape):


        print("shape new",Normal.get_shape().as_list())
        #####time.sleep(10)
        matrix_dimensions = tf.shape(Normal)
        batch_size,matrix_size,hidden_size = matrix_dimensions[0],matrix_dimensions[1],matrix_dimensions[2]


        print(matrix_size)
        print(name)
        #####time.sleep(15)
        Wf = tf.get_variable(name,shape=[4*NormalShape.get_shape().as_list()[1],1])

        nameBias = name+"bias"
        Bias = tf.get_variable(nameBias,shape=[1])

        ElementWiseProduct = tf.multiply(Tilda,Normal)

        SubtractMatrix = tf.subtract(Normal,Tilda)

        ConcatMatrix = tf.concat([tf.concat([tf.concat([Normal,Tilda],1),ElementWiseProduct],1),SubtractMatrix],1) #(batch,4*paragraph,hidden*2)

        print(Normal.get_shape().as_list())
        print(Tilda.get_shape().as_list())
        print(ElementWiseProduct.get_shape().as_list())
        print(SubtractMatrix.get_shape().as_list())
        print(ConcatMatrix.get_shape().as_list())



        ConcatMatrixReshape = tf.reshape(tf.transpose(ConcatMatrix,perm=[1,0,2]),[4*matrix_size,batch_size*hidden_size])

        Matching1 = tf.matmul(tf.transpose(Wf,perm=[1,0]),ConcatMatrixReshape)

        print(Matching1.get_shape().as_list())

        Matching2 = tf.reshape(Matching1,[batch_size,1,hidden_size])


        addBias = tf.add(Matching2,Bias)


        gatingOutput = tf.nn.sigmoid(addBias)

        return gatingOutput

    def Fuse(self,Tilda,Normal,name1,name2,NormalShape):

        gatingOutput = self.gating_function(Tilda,Normal,name1,NormalShape)
        matchingOutput = self.matching_function(Tilda,Normal,name2,NormalShape)

        print(gatingOutput.get_shape().as_list())
        print(matchingOutput.get_shape().as_list())
        ####time.sleep(10)

        returnNewTildaPart1 = tf.matmul(matchingOutput,tf.transpose(gatingOutput,perm=[0,2,1]))

        returnNewTildaPart2 = tf.matmul(matchingOutput,tf.transpose(tf.subtract(1.00,gatingOutput),perm=[0,2,1]))

        returnNewTilda = tf.add(returnNewTildaPart1,returnNewTildaPart2)

        return returnNewTilda

        #returnNewTildaPart1 = tf.matmul(matching_function(Tilda,Normal,name1),tf.transpose(gating_function(Tilda,Normal,name2),perm=[0,2,1]))
        #returnNewTildaPart2 = tf.matmul(Normal,tf.subtract(1,gating_function(Tilda,Normal,name)))
        '''


    def build_graph_middle(self,new_attn,attn_output,context_hiddens,question_hiddens):

        

        matrix_dimensions_answer = context_hiddens.get_shape().as_list()
        batch_size_answer,matrix_size_answer,hidden_size_answer = matrix_dimensions_answer[0],matrix_dimensions_answer[1],matrix_dimensions_answer[2]



        matrix_dimensions_question =  question_hiddens.get_shape().as_list()
        batch_size_question,matrix_size_question,hidden_size_question = matrix_dimensions_question[0],matrix_dimensions_question[1],matrix_dimensions_question[2]



        print(matrix_dimensions_answer,matrix_dimensions_question)
        ##time.sleep(100)

        #Add attention over attention code



        print("question",question_hiddens.get_shape().as_list())
        print("pargraph",context_hiddens.get_shape().as_list())
        print("attention matrix",new_attn.get_shape().as_list())


        P2Q = tf.nn.softmax(new_attn,1)   #(batch,paragraph,questions)

        QTilda = tf.matmul(P2Q,question_hiddens)        #(batch,paragraph,hidden*2) same as paragraph



        Q2P = tf.nn.softmax(new_attn,2)

        Q2PTranspose = tf.transpose(Q2P,perm=[0,2,1])

        PTilda = tf.matmul(Q2PTranspose,context_hiddens)        #(batch,question,hidden*2) same as question


        print("P2Q",P2Q.get_shape().as_list())
        print("QTilda",QTilda.get_shape().as_list())
        print("Q2P",Q2P.get_shape().as_list())
        print("PTilda",PTilda.get_shape().as_list())


        #Fusion layer below

        #variable_temp = self.Fuse(QTilda,context_hiddens,"paragraphGate","paragraphMatch",context_hiddens)
        #print(variable_temp.get_shape().as_list())
        print("AAA")
        ##time.sleep(100)


        paragraphNew = self.Fuse(QTilda,context_hiddens,"paragraphGate","paragraphMatchYOYO",context_hiddens)    #(batch,paragraph,hidden)
        paragraphNew.set_shape([None,matrix_size_answer,hidden_size_answer])
        questionNew = self.Fuse(PTilda,question_hiddens,"questionGate","questionMatch",question_hiddens)      #(batch,question,hidden)
        questionNew.set_shape([None,matrix_size_question,hidden_size_question])
        ##time.sleep(100)

        #paragraphNew = tf.Print(paragraphNew,[tf.shape(paragraphNew)])
        #questionNew = tf.Print(questionNew,[tf.shape(questionNew)])



        print(paragraphNew)
        print(questionNew)

        ##time.sleep(100)


        #paragraphNewMask  = tf.placeholder(tf.int32, shape=[None, 1])
        #questionNewMask  = tf.placeholder(tf.int32, shape=[None, 1])

        encoder2 = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        encoder2Q = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)

        context_hiddens_new = encoder2.build_graph(paragraphNew, self.context_mask,"rnnencoder2")  #(batch,paragraph,context_len)
        question_hiddens_new = encoder2Q.build_graph(questionNew, self.qn_mask,"rnnencoder2Q")   #(batch,question,context_len)

        #context_hiddens_new = paragraphNew
        #question_hiddens_new = questionNew

        #context_hiddens_new = tf.Print(context_hiddens_new,[tf.shape(context_hiddens_new)])
        #question_hiddens_new = tf.Print(question_hiddens_new,[tf.shape(question_hiddens_new)])


        print(context_hiddens_new.get_shape().as_list())
        print("****")
        ####time.sleep(100)
        matrix_dimensions = tf.shape(context_hiddens)
        batch_size,matrix_size,hidden_size = matrix_dimensions[0],matrix_dimensions[1],matrix_dimensions[2]



        #Second fusing layer and softmax layer
        #New learnable matrix



        W1 = tf.get_variable("W1",shape=[matrix_size_answer,matrix_size_answer],trainable=True) #(matrix_size,matrix_size)


        #paragraphNewReshape = tf.reshape(context_hiddens_new,[batch_size*matrix_size,hidden_size])
        paragraphNewTranspose = tf.transpose(context_hiddens_new,perm=[0,2,1])
        paragraphNewReshape = tf.reshape(paragraphNewTranspose,[batch_size*hidden_size,matrix_size])    #(B*H,P)

        paragraphTempRep = tf.matmul(paragraphNewReshape,W1)                                            #(B*H,P)

        paragraphTempRep2 = tf.reshape(paragraphTempRep,[batch_size,hidden_size,matrix_size])
        paragraphTempRep3 = tf.matmul(paragraphTempRep2,context_hiddens_new)
        paragraphTempSoftmax = tf.nn.softmax(paragraphTempRep3)                             #(batch,hidden_size,hidden_size)

        paragraphSelfAllign = tf.matmul(paragraphTempSoftmax,tf.transpose(context_hiddens_new,perm=[0,2,1]))

        paragraphContextual = self.Fuse(tf.transpose(paragraphSelfAllign,perm=[0,2,1]),context_hiddens_new,"paragraphGate2","paragraphMatch2",context_hiddens) #(batch,pargraph,hidden)


        print(paragraphContextual.get_shape().as_list())
        #time.sleep(100)

        #paragraphContextual = tf.Print(paragraphContextual,[tf.shape(paragraphContextual)])


        '''
        batch_size2,matrix_size2,hidden_size2 = matrix_dimensions2[0],matrix_dimensions2[1],matrix_dimensions2[2]
        matrix_dimensions2 = tf.shape(context_hiddens_new)
        questionNewReshape = tf.reshape(question_hiddens_new,[batch_size2*matrix_size2,hidden_size2])
        questionTempRep = tf.matmul(tf.matmul(questionNewReshape,W1))
        questionTempRep2 = tf.reshape(questionTempRep,[batch_size2,matrix_size2,hidden_size2])
        questionTempRep3 = tf.matmul(questionTempRep2,tf.transpose(question_hiddens_new,dim=[0,2,1]))
        questionTempSoftmax = tf.nn.softmax(questionTempRep3)

        questionSelfAllign = tf.matmul(questionTempSoftmax,question_hiddens_new)
        '''

        encoder3 = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)



        #pargraphContextualMask =  tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        paragraphContextual.set_shape([batch_size_answer,matrix_size_answer,hidden_size_answer])

        print(batch_size_answer,matrix_size_answer,hidden_size_answer)
        print(self.context_mask.get_shape().as_list())
        #time.sleep(100)

        paragraphContextual = paragraphContextual
        #paragraphContextual=encoder3.build_graph(paragraphContextual, self.context_mask,"rnnencoder3")  #(batch,paragraph,context_len)


        #Code to represent question
        matrix_dimensions2 = tf.shape(question_hiddens)
        batch_size2,matrix_size2,hidden_size2 = matrix_dimensions2[0],matrix_dimensions2[1],matrix_dimensions2[2]



        #encoder4 = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        #questionSelfAllignMask =  tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        questionSelfAllign = question_hiddens_new
        #encoder4.build_graph(question_hiddens_new, self.qn_mask,"rnnencoder4")  #(batch,question,H)


        Wq = tf.get_variable("Wq",shape=[1,question_hiddens.get_shape().as_list()[2]],trainable=True)      #(1,h)

        questionSelfAllignTranspose = tf.transpose(questionSelfAllign,perm=[2,0,1])
        questionSelfAllignReshape = tf.reshape(questionSelfAllignTranspose,[hidden_size2,matrix_size2*batch_size2])

        GammaTemp = tf.matmul(Wq,questionSelfAllignReshape)
        GammaTemp2 = tf.reshape(GammaTemp,[batch_size2,1,matrix_size2])
        Gamma = tf.nn.softmax(GammaTemp2)      #(batch,1,question)

        questionContextual = tf.matmul(Gamma,questionSelfAllign)    #(batch,1,hidden)

        print(questionContextual.get_shape().as_list())
        ###time.sleep(100)

        #For start point of answer
        WeightSoftmaxStart = tf.get_variable("WeightSoftmaxStart",[question_hiddens.get_shape().as_list()[2],question_hiddens.get_shape().as_list()[2]],trainable=True)
        questionTranspose = tf.transpose(questionContextual,perm=[0,2,1])
        questionContextualReshape = tf.reshape(questionTranspose,[batch_size,hidden_size])
        tempMatrixMult1 = tf.matmul(questionContextualReshape,WeightSoftmaxStart)

        tempMatrixMult1Reshape = tf.reshape(tempMatrixMult1,[batch_size,1,hidden_size])
        probStartMatrix = tf.matmul(tempMatrixMult1Reshape,tf.transpose(paragraphContextual,perm=[0,2,1]))  #(b,1,n)
        '''
        paragraphContextualTranspose = tf.reshape(paragraphContextual,[batch_size*matrix_size,hidden_size])

        tempMatrixMult1 = tf.matmul(paragraphContextualTranspose,WeightSoftmaxStart)
        tempMatrixMult1Reshape = tf.reshape(tempMatrixMult1,[batch_size,matrix_size,1])

        probStartMatrix = tf.matmul(tempMatrixMult1Reshape,questionContextual) #(batch,pargraph,context)
        '''

        #For end point of answer
        WeightSoftmaxEnd = tf.get_variable("WeightSoftmaxEnd",[question_hiddens.get_shape().as_list()[2],question_hiddens.get_shape().as_list()[2]],trainable=True)
        #questionTranspose = tf.transpose(questionContextual,perm=[0,2,1])
        #questionContextualReshape = tf.reshape(questionTranspose,[batch_size,hidden_size])
        tempMatrixMult2 = tf.matmul(questionContextualReshape,WeightSoftmaxEnd)
        tempMatrixMult1Reshape2 = tf.reshape(tempMatrixMult2,[batch_size,1,hidden_size])
        probEndMatrix = tf.matmul(tempMatrixMult1Reshape2,tf.transpose(paragraphContextual,perm=[0,2,1])) #(b,1,n)


        print(probStartMatrix.get_shape().as_list())
        print(probEndMatrix.get_shape().as_list())
        print("**************")


        probStartMatrix = tf.reshape(probStartMatrix,[batch_size,matrix_size])
        probEndMatrix = tf.reshape(probEndMatrix,[batch_size,matrix_size])

        # Concat attn_output to context_hiddens to get blended_reps
        blended_reps = tf.concat([context_hiddens, attn_output], axis=2) # (batch_size, context_len, hidden_size*4)

        # Apply fully connected layer to each blended representation
        # Note, blended_reps_final corresponds to b' in the handout
        # Note, tf.contrib.layers.fully_connected applies a ReLU non-linarity here by default
        blended_reps_final = tf.contrib.layers.fully_connected(blended_reps, num_outputs=self.FLAGS.hidden_size) # blended_reps_final is shape (batch_size, context_len, hidden_size)

        
        return probStartMatrix,probEndMatrix,blended_reps_final
        


        


    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.

        Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
          self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """

        # Use a RNN to get hidden states for the context and the question
        # Note: here the RNNEncoder is shared (i.e. the weights are the same)
        # between the context and the question.

        encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        encoderQ = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        context_hiddens = encoder.build_graph(self.context_embs, self.context_mask,"rnnencoder1") # (batch_size, context_len, hidden_size*2)
        question_hiddens = encoderQ.build_graph(self.qn_embs, self.qn_mask,"rnnencoderQ") # (batch_size, question_len, ,"rnnencoder1"hidden_size*2)

        # Use context hidden states to attend to question hidden states
        attn_layer = BasicAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        _, attn_output,new_attn = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens,2*self.FLAGS.hidden_size) # attn_output is shape (batch_size, context_len, hidden_size*2)

        _,_,blended_reps_final=build_graph_middle(self,new_attn,attn_output,context_hiddens,question_hiddens)

        # Use softmax layer to compute probability distribution for start location
        # Note this produces self.logits_start and self.probdist_start, both of which have shape (batch_size, context_len)
        with vs.variable_scope("StartDist"):
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_final, self.context_mask)

        # Use softmax layer to compute probability distribution for end location
        # Note this produces self.logits_end and self.probdist_end, both of which have shape (batch_size, context_len)
        with vs.variable_scope("EndDist"):
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_final, self.context_mask)

        

        '''
        
        '''
    def add_loss(self):
        """
        Add loss computation to the graph.

        Uses:
          self.logits_start: shape (batch_size, context_len)
            IMPORTANT: Assumes that self.logits_start is masked (i.e. has -large in masked locations).
            That's because the tf.nn.sparse_softmax_cross_entropy_with_logits
            function applies softmax and then computes cross-entropy loss.
            So you need to apply masking to the logits (by subtracting large
            number in the padding location) BEFORE you pass to the
            sparse_softmax_cross_entropy_with_logits function.

          self.ans_span: shape (batch_size, 2)
            Contains the gold start and end locations

        Defines:
          self.loss_start, self.loss_end, self.loss: all scalar tensors
        """
        with vs.variable_scope("loss"):

            # Calculate loss for prediction of start position
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0]) # loss_start has shape (batch_size)
            self.loss_start = tf.reduce_mean(loss_start) # scalar. avg across batch
            tf.summary.scalar('loss_start', self.loss_start) # log to tensorboard

            # Calculate loss for prediction of end position
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
            self.loss_end = tf.reduce_mean(loss_end)
            tf.summary.scalar('loss_end', self.loss_end)

            # Add the two losses
            self.loss = self.loss_start + self.loss_end
            tf.summary.scalar('loss', self.loss)


    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        #print(batch.context_ids)
        input_feed[self.context_mask] = batch.context_mask
        #print(batch.context_mask)
        input_feed[self.qn_ids] = batch.qn_ids
        #print(batch.qn_ids)
        input_feed[self.qn_mask] = batch.qn_mask
        #print(batch.qn_mask)
        input_feed[self.ans_span] = batch.ans_span
        #print(batch.ans_span)
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout

        
        # print(len(input_feed))
        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        #print(session.run(output_feed, input_feed))
        session.run(tf.global_variables_initializer())
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed,run_metadata=None)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss


    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          probdist_start and probdist_end: both shape (batch_size, context_len)
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout
        session.run(tf.global_variables_initializer())
        output_feed = [self.probdist_start, self.probdist_end]
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end


    def get_start_end_pos(self, session, batch):
        """
        Run forward-pass only; get the most likely answer span.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          start_pos, end_pos: both numpy arrays shape (batch_size).
            The most likely start and end positions for each example in the batch.
        """
        # Get start_dist and end_dist, both shape (batch_size, context_len)
        start_dist, end_dist = self.get_prob_dists(session, batch)

        # Take argmax to get start_pos and end_post, both shape (batch_size)
        start_pos = np.argmax(start_dist, axis=1)
        end_pos = np.argmax(end_dist, axis=1)

        return start_pos, end_pos


    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):
        """
        Get loss for entire dev set.

        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files

        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        
        for batch in get_batch_generator(self.word2id, dev_context_path, dev_qn_path, dev_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

            # Get loss for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print("Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic))

        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss


    def check_f1_em(self, session, context_path, qn_path, ans_path, dataset, num_samples=100, print_to_screen=True):
        """
        Sample from the provided (train/dev) set.
        For each sample, calculate F1 and EM score.
        Return average F1 and EM score for all samples.
        Optionally pretty-print examples.

        Note: This function is not quite the same as the F1/EM numbers you get from "official_eval" mode.
        This function uses the pre-processed version of the e.g. dev set for speed,
        whereas "official_eval" mode uses the original JSON. Therefore:
          1. official_eval takes your max F1/EM score w.r.t. the three reference answers,
            whereas this function compares to just the first answer (which is what's saved in the preprocessed data)
          2. Our preprocessed version of the dev set is missing some examples
            due to tokenization issues (see squad_preprocess.py).
            "official_eval" includes all examples.

        Inputs:
          session: TensorFlow session
          qn_path, context_path, ans_path: paths to {dev/train}.{question/context/answer} data files.
          dataset: string. Either "train" or "dev". Just for logging purposes.
          num_samples: int. How many samples to use. If num_samples=0 then do whole dataset.
          print_to_screen: if True, pretty-prints each example to screen

        Returns:
          F1 and EM: Scalars. The average across the sampled examples.
        """
        logging.info("Calculating F1/EM for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))

        f1_total = 0.
        em_total = 0.
        example_num = 0

        tic = time.time()

    
        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=False):

            pred_start_pos, pred_end_pos = self.get_start_end_pos(session, batch)

            # Convert the start and end positions to lists length batch_size
            pred_start_pos = pred_start_pos.tolist() # list length batch_size
            pred_end_pos = pred_end_pos.tolist() # list length batch_size

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1

                # Get the predicted answer
                # Important: batch.context_tokens contains the original words (no UNKs)
                # You need to use the original no-UNK version when measuring F1/EM
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens)

                # Calc F1/EM
                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                # Optionally pretty-print
                
                if print_to_screen:
                    print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx], batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em)
                
                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total


    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        """
        Main training loop.

        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in get_batch_generator(self.word2id, train_context_path, train_qn_path, train_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    # Get F1/EM on train set and log to tensorboard
                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    # Get F1/EM on dev set and log to tensorboard
                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)


                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    if best_dev_em is None or dev_em > best_dev_em:
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()



def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
