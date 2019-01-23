

"""This file contains some basic model components"""
import time
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    ------Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size,kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size,kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks,name):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.++
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope(name):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out




class RNNEncoderLSTM(object):
   

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks,name):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.++
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope(name):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)





            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)

            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist



class SimpleSoftmaxLayerNew(object):
   
    def __init__(self):
        pass

    def build_graph(self, logits, masks):
     
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            #logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)

            #logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist




class BasicAttn(object):
   

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:++
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys,hidden_len):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.+
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):


            print(values.get_shape().as_list())
            ##time.sleep(100)
            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, hidden, num_values)

            WLin = tf.get_variable("WLin", [hidden_len,hidden_len],trainable=True)

            print("WLin",WLin.get_shape().as_list())

            values_t_transpose = tf.transpose(values,perm=[0,2,1])          #Create (batch,hidden,questions)
            values_t_transpose_2 = tf.transpose(values_t_transpose,perm=[1,0,2])     #transpose it so can be multiplied with (2h,2h)

            dimension_values_t = tf.shape(values)

            dimension_row = dimension_values_t[2]
            dimension_col = dimension_values_t[0] * dimension_values_t[1]




            print(values_t_transpose_2.get_shape().as_list())
            #time.sleep(100)
            values_t_reshape = tf.reshape(values_t_transpose_2,[dimension_row,dimension_col]) #Multuply this with (2h,2h) #Its size is (hidden,batch*question)



            Multiply1 = tf.matmul(tf.transpose(WLin),values_t_reshape)


            Multiply1_reshape_new = tf.reshape(Multiply1,[dimension_values_t[2],dimension_values_t[0],dimension_values_t[1]]) 
            Multiply1_reshape = tf.transpose(Multiply1_reshape_new,perm=[1,0,2]) #(batch,2h,questions)


            keys_transpose = tf.transpose(keys,perm=[0,2,1])          #Create (batch,hidden,paragraph)
            keys_transpose_2 = tf.transpose(keys_transpose,perm=[1,0,2])     #transpose it so can be multiplied with (2h,2h)

            dimension_values_keys = [tf.shape(keys)[0],tf.shape(keys)[1],tf.shape(keys)[2]]
            dimension_row = dimension_values_keys[2]
            dimension_col = dimension_values_keys[0]* dimension_values_keys[1]

            keys_reshape = tf.reshape(keys_transpose_2,[dimension_row,dimension_col]) #Multuply this with (2h,2h)

            Multiply2 = tf.matmul(WLin,keys_reshape)

            Multiply2_reshape_new = tf.reshape(Multiply2,[dimension_values_keys[2],dimension_values_keys[0],dimension_values_keys[1]])
            Multiply2_reshape = tf.transpose(Multiply2_reshape_new,perm=[1,0,2]) #(batch,2h,paragraph)


            Multiply_attention = tf.matmul(tf.nn.relu(tf.transpose(Multiply1_reshape,perm=[0,2,1])),tf.nn.relu(Multiply2_reshape))

            Multiply_attention_order = tf.transpose(Multiply_attention,perm=[0,2,1]) #(batch,paragraph,questions)



            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)

            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)

            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)



            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output, Multiply_attention_order


def masked_softmax(logits, mask, dim):
    
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
