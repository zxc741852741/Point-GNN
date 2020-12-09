"""This file defines classes for the graph neural network. """

from functools import partial

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import sys

def instance_normalization(features):
    with tf.variable_scope(None, default_name='IN'):
        mean, variance = tf.nn.moments(
            features, [0], name='IN_stats', keep_dims=True)
        features = tf.nn.batch_normalization(
            features, mean, variance, None, None, 1e-12, name='IN_apply')
    return(features)

normalization_fn_dict = {
    'fused_BN_center': slim.batch_norm,
    'BN': partial(slim.batch_norm, fused=False, center=False),
    'BN_center': partial(slim.batch_norm, fused=False),
    'IN': instance_normalization,
    'NONE': None
}
activation_fn_dict = {
    'ReLU': tf.nn.relu,
    'ReLU6': tf.nn.relu6,
    'LeakyReLU': partial(tf.nn.leaky_relu, alpha=0.01),
    'ELU':tf.nn.elu,
    'NONE': None,
    'Sigmoid': tf.nn.sigmoid,
    'Tanh': tf.nn.tanh,
}

def multi_layer_fc_fn(sv, mask=None, Ks=(64, 32, 64), num_classes=4,
    is_logits=False, num_layer=4, normalization_type="fused_BN_center",
    activation_type='ReLU'):
    """A function to create multiple layers of neural network to compute
    features passing through each edge.

    Args:
        sv: a [N, M] or [T, DEGREE, M] tensor.
        N is the total number of edges, M is the length of features. T is
        the number of recieving vertices, DEGREE is the in-degree of each
        recieving vertices. When a [T, DEGREE, M] tensor is provided, the
        degree of each recieving vertex is assumed to be same.
        N is the total number of edges, M is the length of features. T is
        the number of recieving vertices, DEGREE is the in-degree of each
        recieving vertices. When a [T, DEGREE, M] tensor is provided, the
        degree of each recieving vertex is assumed to be same.
        mask: a optional [N, 1] or [T, DEGREE, 1] tensor. A value 1 is used
        to indicate a valid output feature, while a value 0 indicates
        an invalid output feature which is set to 0.
        num_layer: number of layers to add.

    returns: a [N, K] tensor or [T, DEGREE, K].
        K is the length of the new features on the edge.
    """
    assert len(sv.shape) == 2
    assert len(Ks) == num_layer-1
    if is_logits:
        features = sv
        for i in range(num_layer-1):
            features = slim.fully_connected(features, Ks[i],
                activation_fn=activation_fn_dict[activation_type],
                normalizer_fn=normalization_fn_dict[normalization_type],
                )
        features = slim.fully_connected(features, num_classes,
            activation_fn=None,
            normalizer_fn=None
            )
    else:
        features = sv
        for i in range(num_layer-1):
            features = slim.fully_connected(features, Ks[i],
                activation_fn=activation_fn_dict[activation_type],
                normalizer_fn=normalization_fn_dict[normalization_type],
                )
        features = slim.fully_connected(features, num_classes,
            activation_fn=activation_fn_dict[activation_type],
            normalizer_fn=normalization_fn_dict[normalization_type],
            )
    if mask is not None:
        features = features * mask
    return features

def multi_layer_neural_network_fn(features, Ks=(64, 32, 64), is_logits=False,
    normalization_type="fused_BN_center", activation_type='ReLU'):
    """A function to create multiple layers of neural network.
    """
    assert len(features.shape) == 2
    if is_logits:
        for i in range(len(Ks)-1):
            features = slim.fully_connected(features, Ks[i],
                activation_fn=activation_fn_dict[activation_type],
                normalizer_fn=normalization_fn_dict[normalization_type])
        features = slim.fully_connected(features, Ks[-1],
            activation_fn=None,
            normalizer_fn=None)
    else:
        for i in range(len(Ks)):
            features = slim.fully_connected(features, Ks[i],
                activation_fn=activation_fn_dict[activation_type],
                normalizer_fn=normalization_fn_dict[normalization_type])
    return features

def graph_scatter_max_fn(point_features, point_centers, num_centers):
    aggregated = tf.math.unsorted_segment_max(point_features,
        point_centers, num_centers, name='scatter_max')
    return aggregated

def graph_scatter_sum_fn(point_features, point_centers, num_centers):
    aggregated = tf.math.unsorted_segment_sum(point_features,
        point_centers, num_centers, name='scatter_sum')
    return aggregated

def graph_scatter_mean_fn(point_features, point_centers, num_centers):
    aggregated = tf.math.unsorted_segment_mean(point_features,
        point_centers, num_centers, name='scatter_mean')
    return aggregated

class ClassAwarePredictor(object):
    """A class to predict 3D bounding boxes and class labels."""

    def __init__(self, cls_fn, loc_fn):
        """
        Args:
            cls_fn: a function to classify labels.
            loc_fn: a function to predict 3D bounding boxes.
        """
        self._cls_fn = cls_fn
        self._loc_fn = loc_fn

    def apply_regular(self, features, num_classes, box_encoding_len,
        normalization_type='fused_BN_center',
        activation_type='ReLU'):
        """
        Args:
            input_v: input feature vectors. [N, M].
            output_v: not used.
            A: not used.
            num_classes: the number of classes to predict.

        returns: logits, box_encodings.
        """
        box_encodings_list = []
        with tf.variable_scope('predictor'):
            with tf.variable_scope('cls'):
                logits = self._cls_fn(
                    features, num_classes=num_classes, is_logits=True,
                    normalization_type=normalization_type,
                    activation_type=activation_type)
            with tf.variable_scope('loc'):
                for class_idx in range(num_classes):
                    with tf.variable_scope('cls_%d' % class_idx):
                        box_encodings = self._loc_fn(
                            features, num_classes=box_encoding_len,
                            is_logits=True,
                            normalization_type=normalization_type,
                            activation_type=activation_type)
                        box_encodings = tf.expand_dims(box_encodings, axis=1)
                        box_encodings_list.append(box_encodings)
            box_encodings = tf.concat(box_encodings_list, axis=1)
        return logits, box_encodings

class ClassAwareSeparatedPredictor(object):
    """A class to predict 3D bounding boxes and class labels."""

    def __init__(self, cls_fn, loc_fn):
        """
        Args:
            cls_fn: a function to classify labels.
            loc_fn: a function to predict 3D bounding boxes.
        """
        self._cls_fn = cls_fn
        self._loc_fn = loc_fn

    def apply_regular(self, features, num_classes, box_encoding_len,
        normalization_type='fused_BN_center',
        activation_type='ReLU'):
        """
        Args:
            input_v: input feature vectors. [N, M].
            output_v: not used.
            A: not used.
            num_classes: the number of classes to predict.

        returns: logits, box_encodings.
        """
        box_encodings_list = []
        with tf.variable_scope('predictor'):
            with tf.variable_scope('cls'):
                logits = self._cls_fn(
                    features, num_classes=num_classes, is_logits=True,
                    normalization_type=normalization_type,
                    activation_type=activation_type)
            features_splits = tf.split(features, num_classes, axis=-1)
            with tf.variable_scope('loc'):
                for class_idx in range(num_classes):
                    with tf.variable_scope('cls_%d' % class_idx):
                        box_encodings = self._loc_fn(
                            features_splits[class_idx],
                            num_classes=box_encoding_len,
                            is_logits=True,
                            normalization_type=normalization_type,
                            activation_type=activation_type)
                        box_encodings = tf.expand_dims(box_encodings, axis=1)
                        box_encodings_list.append(box_encodings)
            box_encodings = tf.concat(box_encodings_list, axis=1)
        return logits, box_encodings

class PointSetPooling(object):
    """A class to implement local graph netural network."""

    def __init__(self,
        point_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        output_fn=multi_layer_neural_network_fn):
        self._point_feature_fn = point_feature_fn
        self._aggregation_fn = aggregation_fn
        self._output_fn = output_fn

    def apply_regular(self,
        point_features,
        point_coordinates,
        keypoint_indices,
        set_indices,
        edgetozero,                                                                   ## add change
        t_num_group,
        point_MLP_depth_list=None,
        point_MLP_normalization_type='fused_BN_center',
        point_MLP_activation_type = 'ReLU',
        output_MLP_depth_list=None,
        output_MLP_normalization_type='fused_BN_center',
        output_MLP_activation_type = 'ReLU'):
        """apply a features extraction from point sets.

        Args:
            point_features: a [N, M] tensor. N is the number of points.
            M is the length of the features.
            point_coordinates: a [N, D] tensor. N is the number of points.
            D is the dimension of the coordinates.
            keypoint_indices: a [K, 1] tensor. Indices of K keypoints.
            set_indices: a [S, 2] tensor. S pairs of (point_index, set_index).
            i.e. (i, j) indicates point[i] belongs to the point set created by
            grouping around keypoint[j].
            point_MLP_depth_list: a list of MLP units to extract point features.
            point_MLP_normalization_type: the normalization function of MLP.
            point_MLP_activation_type: the activation function of MLP.
            output_MLP_depth_list: a list of MLP units to embedd set features.
            output_MLP_normalization_type: the normalization function of MLP.
            output_MLP_activation_type: the activation function of MLP.

        returns: a [K, output_depth] tensor as the set feature.
        Output_depth depends on the feature extraction options that
        are selected.
        """
        # Gather the points in a set
        point_set_features = tf.gather(point_features, set_indices[:,0])
        point_set_coordinates = tf.gather(point_coordinates, set_indices[:,0])
        # Gather the keypoints for each set
        point_set_keypoint_indices = tf.gather(
            keypoint_indices, set_indices[:, 1])
        point_set_keypoint_coordinates = tf.gather(point_coordinates,
            point_set_keypoint_indices[:,0])
        # points within a set use relative coordinates to its keypoint
        point_set_coordinates = \
            point_set_coordinates - point_set_keypoint_coordinates
        point_set_features = tf.concat(
            [point_set_features, point_set_coordinates], axis=-1)
        print('point_set_features = {}'.format(point_set_features))
        with tf.variable_scope('extract_vertex_features'):
            # Step 1: Extract all vertex_features
            extracted_point_features = self._point_feature_fn(
                point_set_features,
                Ks=point_MLP_depth_list, is_logits=False,
                normalization_type=point_MLP_normalization_type,
                activation_type=point_MLP_activation_type)
            set_features = self._aggregation_fn(
                extracted_point_features, set_indices[:, 1],
                tf.shape(keypoint_indices)[0])
        with tf.variable_scope('combined_features'):
            set_features = self._output_fn(set_features,
                Ks=output_MLP_depth_list, is_logits=False,
                normalization_type=output_MLP_normalization_type,
                activation_type=output_MLP_activation_type)
        return set_features
def attention_v2(edge_features,edges,t_num_group):
    def get_duplicated(tensor):
        unique_a_vals, unique_idx = tf.unique(tensor)
        #init = tf.initialize_all_variables() 
        count_a_unique = tf.unsorted_segment_sum(tf.ones_like(tensor),unique_idx,tf.shape(tensor)[0])                   
        #more_than_one = tf.greater(count_a_unique, 1)                               
        #more_than_one_idx = tf.squeeze(tf.where(more_than_one))                     
        #more_than_one_vals = tf.squeeze(tf.gather(unique_a_vals, more_than_one_idx)) #[not_duplicated, _] 
        not_duplicated,_= tf.raw_ops.ListDiff(x = tensor, y = unique_a_vals, out_idx=tf.dtypes.int32, name=None)                
        dups_in_a, indexes_in_a = tf.raw_ops.ListDiff(x = tensor, y =not_duplicated,out_idx=tf.dtypes.int32, name=None)
        return dups_in_a, indexes_in_a
    with tf.device("/cpu:0"):
    #aaa=1
    #if aaa ==1:
        with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        #if aaa==1:
            #for i,local_edge_features in enumerate(dd):
            #reshaped_o = tf.reshape(tf.cast(local_edge_features, tf.uint8), [-1, 1, 1])
            #not_empty_ops.append(tf.logical_or(i>max_idx, tf.size(reshaped_o)>0))
            #png_output = tf.cond(tf.size(reshaped_o)>0, lambda : att(local_edge_features), lambda: 1.0)
            #print(tf.shape(png_output))
            #print(type(png_output))
            #att(local_edge_features)
            #print(i)
            #local_edge_features = tf.gather(edge_features,idx_list)
            '''Q = slim.fully_connected(edge_features, 300,
                activation_fn=None,
                normalizer_fn=None)
            K = slim.fully_connected(edge_features, 300,
                activation_fn=None,
                normalizer_fn=None)
            V = slim.fully_connected(edge_features, 300,
                activation_fn=None,
                normalizer_fn=None)'''
            #output = tf.keras.layers.Attention()([Q, K,V])

            #return output
            '''denominator = tf.sqrt(tf.cast(tf.shape(Q)[0],tf.float32), name=None) 
            molecular = tf.tensordot(Q,tf.transpose(K),axes = 1)
            QK = molecular/denominator
            alphas = tf.nn.softmax(QK, name='alphas')    
            output = tf.tensordot(alphas,V,axes=1)
            print('output = {}'.format(output))
            print(type(output))
            return output'''

            Q = tf.layers.dense(edge_features, 300, use_bias=True) # (N, T_q, d_model)
            K = tf.layers.dense(edge_features,  300, use_bias=True) # (N, T_k, d_model)
            V = tf.layers.dense(edge_features, 300, use_bias=True) # (N, T_k, d_model)
            #tf.cast(x, dtype)
            a = edges[:, 1]
            dups_in_a, indexes_in_a = get_duplicated(a)
            break_point = tf.unsorted_segment_sum(tf.ones_like(a),                   
                                                 a,                        
                                                 tf.shape(a)[0])
            '''dups_in_Q, indexes_in_Q = get_duplicated(Q)
            break_point_Q = tf.unsorted_segment_sum(tf.ones_like(Q),                   
                                                 Q,                        
                                                 tf.shape(Q)[0])'''
            '''dups_in_K, indexes_in_K = get_duplicated(K)
            break_point_K = tf.unsorted_segment_sum(tf.ones_like(K),                   
                                                 K,                        
                                                 tf.shape(K)[0])
            dups_in_V, indexes_in_V = get_duplicated(V)
            break_point_V = tf.unsorted_segment_sum(tf.ones_like(V),                   
                                                 V,                        
                                                 tf.shape(V)[0])'''
            QQ = tf.dynamic_partition(Q, dups_in_a, num_partitions=3500)
            KK = tf.dynamic_partition(K, dups_in_a, num_partitions=3500)
            VV = tf.dynamic_partition(V, dups_in_a, num_partitions=3500)


            #print('QQ = {}'.format(QQ))
            print('QQ = {}'.format(type(QQ)))
            #------------------------------------------------
            new_edge_feature = []

            for i,(local_QQ,local_KK,local_VV) in enumerate(zip(QQ,KK,VV)):

            #molecular = tf.matmul(QQ,KK)

            #print('molecular = {}'.format(molecular))
            #QK = molecular

                output = tf.keras.layers.Attention()([local_QQ, local_KK,local_VV])

                #print('Q_SHAPE = {}'.format(Q.get_shape().as_list()))
                #print('k_SHAPE = {}'.format(K.get_shape().as_list()))
                #print('v_SHAPE = {}'.format(V.get_shape().as_list()))
                #print(K.get_shape().as_list())
                #print(V.get_shape().as_list())
                '''denominator = tf.sqrt(tf.cast(tf.shape(local_QQ)[0],tf.float32), name=None) 
                molecular = tf.tensordot(local_QQ,tf.transpose(local_KK),axes = 1)
                QK = molecular/denominator
                alphas = tf.nn.softmax(QK, name='alphas')    
                output = tf.tensordot(alphas,local_VV,axes=1)'''

                new_edge_feature.append(output)
                print('output = {}'.format(output))
                print(type(output))
                #return output
            concat_tensor = tf.concat(new_edge_feature, 0)
            #


            concat_tensor = tf.layers.dense(concat_tensor, 300, use_bias=True)
            return concat_tensor
    #------------------------------------------------need modify

def attention(edge_features,edges,t_num_group):
    def dynamic_partition_png(vals, idx, max_partitions):
        """Encodes output of dynamic partition as a Tensor of png-encoded strings."""
        max_idx = tf.reduce_max(idx)
        max_vals = tf.reduce_max(idx)
        with tf.control_dependencies([tf.Assert(max_vals<256, ["vals must be <256"])]):
            outputs = tf.dynamic_partition(vals, idx, num_partitions=max_partitions)
        png_outputs = []
        dummy_png = tf.image.encode_png(([[[2]]]))
        not_empty_ops = [] # ops that detect empty lists that aren't at the end
        for i, o in enumerate(outputs):
            reshaped_o = tf.reshape(tf.cast(o, tf.uint8), [-1, 1, 1])

            png_output = tf.cond(tf.size(reshaped_o)>0, lambda: tf.image.encode_png(reshaped_o), lambda: dummy_png)
            png_outputs.append(png_output)
            not_empty_ops.append(tf.logical_or(i>max_idx, tf.size(reshaped_o)>0))
        packed_tensor = tf.stack(png_outputs)
        no_illegal_empty_lists = tf.reduce_all(tf.stack(not_empty_ops))
        with tf.control_dependencies([tf.Assert(no_illegal_empty_lists, ["empty lists must be last"])]):
            result = packed_tensor[:max_idx+1]
        return result

    def decode(p):
        return tf.image.decode_png(p)[:, 0, 0]
    def get_duplicated(tensor):
        unique_a_vals, unique_idx = tf.unique(tensor)
        #init = tf.initialize_all_variables() 
        count_a_unique = tf.unsorted_segment_sum(tf.ones_like(tensor),unique_idx,tf.shape(tensor)[0])                   
        #more_than_one = tf.greater(count_a_unique, 1)                               
        #more_than_one_idx = tf.squeeze(tf.where(more_than_one))                     
        #more_than_one_vals = tf.squeeze(tf.gather(unique_a_vals, more_than_one_idx)) #[not_duplicated, _] 
        not_duplicated,_= tf.raw_ops.ListDiff(x = tensor, y = unique_a_vals, out_idx=tf.dtypes.int32, name=None)                
        dups_in_a, indexes_in_a = tf.raw_ops.ListDiff(x = tensor, y =not_duplicated,out_idx=tf.dtypes.int32, name=None)
        return dups_in_a, indexes_in_a

    #for i in range(len(Ks)-1):
    
    #print(tf.shape(edge_features))
    '''Q = slim.fully_connected(edge_features,300,
        activation_fn=None,
        normalizer_fn=None)
    K = slim.fully_connected(edge_features,300,
        activation_fn=None,
        normalizer_fn=None)
    V = slim.fully_connected(edge_features, 300,
        activation_fn=None,
        normalizer_fn=None)'''
    #print(type(edges[:, 1]))
    #sess=tf.Session()
    #t_num_group
    #loop = tf.math.reduce_sum(t_num_group)
    a = edges[:, 1]
    dups_in_a, indexes_in_a = get_duplicated(a)
    break_point = tf.unsorted_segment_sum(tf.ones_like(a),                   
                                         a,                        
                                         tf.shape(a)[0])
    '''i=1
    #tf.cond(tf.size(reshaped_o)>0, lambda : att(local_edge_features), lambda: 1.0)
    an = dups_in_a[0]
    def cond(dups_in_a,i,an):

        return i<10

    def body(dups_in_a,i,an):
        return dups_in_a[i]
    an = tf.while_loop(cond, body, [dups_in_a,i,an])
    print(an,i)'''

    #new_edge_features = tf.gather(edge_features,indexes_in_a)
    #max_value = tf.math.reduce_max(dups_in_a)
    #result = tf.dynamic_partition(new_edge_features, dups_in_a)
    #split_edge_features = tf.split(a,break_point)

    #sess = tf.Session()
    #vals = tf.constant([1,2,3,4,5])
    #idx = [0, 1, 1, 1, 1]
    dd = tf.dynamic_partition(edge_features, dups_in_a, num_partitions=3000)

    #tf_vals = dynamic_partition_png(edge_features, dups_in_a, 3000)
    #print(type(tf_vals))
    #local_edge_features = tf_vals
    def att(local_edge_features):
        with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
            Q = tf.layers.dense(local_edge_features, 300, use_bias=True) # (N, T_q, d_model)
            K = tf.layers.dense(local_edge_features, 300, use_bias=True) # (N, T_k, d_model)
            V = tf.layers.dense(local_edge_features, 300, use_bias=True) # (N, T_k, d_model)
            #tf.cast(x, dtype)
            #print('Q_SHAPE = {}'.format(Q.get_shape().as_list()))
            #print('k_SHAPE = {}'.format(K.get_shape().as_list()))
            #print('v_SHAPE = {}'.format(V.get_shape().as_list()))
            #print(K.get_shape().as_list())
            #print(V.get_shape().as_list())
            denominator = tf.sqrt(tf.cast(tf.shape(local_edge_features)[0],tf.float32), name=None) 
            molecular = tf.tensordot(Q,tf.transpose(K),axes = 1)
            QK = molecular/denominator
            alphas = tf.nn.softmax(QK, name='alphas')    
            output = tf.tensordot(alphas,V,axes=1)
            #print('output = {}'.format(output))
            #print(type(output))
            return output
    max_idx = tf.reduce_max(dups_in_a)
    not_empty_ops = []
    #y = tf.Variable(np.empty((, 300), dtype=np.float32))
    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        for i,local_edge_features in enumerate(dd):
        #reshaped_o = tf.reshape(tf.cast(local_edge_features, tf.uint8), [-1, 1, 1])
        #not_empty_ops.append(tf.logical_or(i>max_idx, tf.size(reshaped_o)>0))
        #png_output = tf.cond(tf.size(reshaped_o)>0, lambda : att(local_edge_features), lambda: 1.0)
        #print(tf.shape(png_output))
        #print(type(png_output))
        #att(local_edge_features)
        #print(i)
        #local_edge_features = tf.gather(edge_features,idx_list)
        
            Q = tf.layers.dense(local_edge_features, 300, use_bias=True) # (N, T_q, d_model)
            K = tf.layers.dense(local_edge_features, 300, use_bias=True) # (N, T_k, d_model)
            V = tf.layers.dense(local_edge_features, 300, use_bias=True) # (N, T_k, d_model)
            #tf.cast(x, dtype)
            print('Q_SHAPE = {}'.format(Q.get_shape().as_list()))
            print('k_SHAPE = {}'.format(K.get_shape().as_list()))
            print('v_SHAPE = {}'.format(V.get_shape().as_list()))
            #print(K.get_shape().as_list())
            #print(V.get_shape().as_list())
            denominator = tf.sqrt(tf.cast(tf.shape(local_edge_features)[0],tf.float32), name=None) 
            molecular = tf.tensordot(Q,tf.transpose(K),axes = 1)
            QK = molecular/denominator
            alphas = tf.nn.softmax(QK, name='alphas')    
            output = tf.tensordot(alphas,V,axes=1)
            print('output = {}'.format(output))
            print(type(output))

            #tf.concat([t1, t2], 0)
        #edge_features[idx_list] = output
        #i+=1
        #pass
    #sess = tf.Session()
    #print(sess.run(dd[0]))
    #print(dd)
    '''def cond(dd):
    return dd

    def body(dd):
        i = i + 1
        return i, n 
    tf.while_loop(cond, body, dd)'''
    '''print('dkfjdslk')
    print(type(dd[0]))
    print(sess.run(dd[0][]))
    tf_vals = dynamic_partition_png(vals, idx, 1000)
    print(tf_vals)
    print(sess.run(decode(tf_vals[0]))) # => [1 2]
    print(sess.run(decode(tf_vals[1]))) # => [3 4 5]'''

    '''i = 0
    break_p = 0
    with tf.Session() as sess:
        scalar = loop.eval()
    for i in scalar:
        pass'''
    '''while(1):

        break_p = break_p + break_point[i]
        indexes_in_a[break_p]
        i+=1'''
    
    #[i for i in dups_in_a]
    #dups_in_a[:6]
    #tf.map_fn(fn=lambda t: [i for i in t], elems=dups_in_a)
    #dups_in_a_2, indexes_in_a_2 = get_duplicated(dups_in_a)
    #indexes_in_a_2
    '''unique_a_vals, unique_idx = tf.unique(edges[:, 1])
    count_a_unique = tf.unsorted_segment_sum(tf.ones_like(a),unique_idx,tf.shape(a)[0])                   
    more_than_one = tf.greater(count_a_unique, 1)                               
    more_than_one_idx = tf.squeeze(tf.where(more_than_one))                     
    more_than_one_vals = tf.squeeze(tf.gather(unique_a_vals, more_than_one_idx)) #[not_duplicated, _] 
    not_duplicated,_= tf.raw_ops.ListDiff(
    x = a, y = more_than_one_vals, out_idx=tf.dtypes.int32, name=None
)
    init = tf.initialize_all_variables()                
    dups_in_a, indexes_in_a = tf.raw_ops.ListDiff(x = a, y =not_duplicated,out_idx=tf.dtypes.int32, name=None)     #'''

    print('------------------')
    print(type(indexes_in_a))
    print('---------------')
    '''with tf.Session() as s:                                                     
        s.run(init)                                                             
        a, dupvals, dupidxes, dia = s.run([a, more_than_one_vals,                    
                                      indexes_in_a, dups_in_a])'''
    '''    print('--------------------') 
        print ("Input: ", a  )                                                    
        print ("Duplicate values: ", dupvals    )                                 
        print ("Indexes of duplicates in a: ", dupidxes)
        print ("Dup vals with dups: ", dia)
        print('--------------------') '''
    #shape = tf.scan(fn, tf.shape(y))
    #loop = tf.size(y)
    #print(shape.eval(session=sess))
    #y = y.eval(session=sess)
    #v = y.get_shape()
    #loop = v.num_elements()
    #print(type(tf.shape(y)[0]))
    #for i in range(sess.run(loop)):
    #edges_idx_array = tf.Session().run(edges[:, 1])
    '''while(1):
        idx_list = [idx for idx,dest in enumerate(edges_idx_array) if dest==y[i]]
    #for i in range(len(y)):
        local_edge_features = tf.gather(edge_features,idx_list)
        Q = tf.layers.dense(local_edge_features, 300, use_bias=True) # (N, T_q, d_model)
        K = tf.layers.dense(local_edge_features, 300, use_bias=True) # (N, T_k, d_model)
        V = tf.layers.dense(local_edge_features, 300, use_bias=True) # (N, T_k, d_model)
        #tf.cast(x, dtype)
        print('Q_SHAPE = {}'.format(Q.get_shape().as_list()))
        print('k_SHAPE = {}'.format(K.get_shape().as_list()))
        print('v_SHAPE = {}'.format(V.get_shape().as_list()))
        #print(K.get_shape().as_list())
        #print(V.get_shape().as_list())
        denominator = tf.sqrt(tf.cast(tf.shape(local_edge_features)[0],tf.float32), name=None) 
        molecular = tf.tensordot(Q,tf.transpose(K),axes = 1)
        QK = molecular/denominator
        alphas = tf.nn.softmax(QK, name='alphas')    
        output = tf.tensordot(alphas,V,axes=1)
        edge_features[idx_list] = output
        i+=1
    return edge_features'''
class GraphNetAutoCenter(object):
    """A class to implement point graph netural network layer."""

    def __init__(self,
        edge_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        update_fn=multi_layer_neural_network_fn,
        auto_offset_fn=multi_layer_neural_network_fn):
        self._edge_feature_fn = edge_feature_fn
        self._aggregation_fn = aggregation_fn
        self._update_fn = update_fn
        self._auto_offset_fn = auto_offset_fn

    def apply_regular(self,
        input_vertex_features,
        input_vertex_coordinates,
        NOT_USED,
        edges,
        edgetozero,                                                      ## add change
        t_num_group,
        edge_MLP_depth_list=None,
        edge_MLP_normalization_type='fused_BN_center',
        edge_MLP_activation_type = 'ReLU',
        update_MLP_depth_list=None,
        update_MLP_normalization_type='fused_BN_center',
        update_MLP_activation_type = 'ReLU',
        auto_offset=False,
        auto_offset_MLP_depth_list=None,
        auto_offset_MLP_normalization_type='fused_BN_center',
        auto_offset_MLP_feature_activation_type = 'ReLU',
        ):
        """apply one layer graph network on a graph. .

        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
            edge_MLP_depth_list: a list of MLP units to extract edge features.
            edge_MLP_normalization_type: the normalization function of MLP.
            edge_MLP_activation_type: the activation function of MLP.
            update_MLP_depth_list: a list of MLP units to extract update
            features.
            update_MLP_normalization_type: the normalization function of MLP.
            update_MLP_activation_type: the activation function of MLP.
            auto_offset: boolean, use auto registration or not.
            auto_offset_MLP_depth_list: a list of MLP units to compute offset.
            auto_offset_MLP_normalization_type: the normalization function.
            auto_offset_MLP_feature_activation_type: the activation function.

        returns: a [N, M] tensor. Updated vertex features.
        """
        # Gather the source vertex of the edges
        s_vertex_features = tf.gather(input_vertex_features, edges[:,0])
        s_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:,0])
        # [optional] Compute the coordinates offset
        if auto_offset:
            offset = self._auto_offset_fn(input_vertex_features,
                Ks=auto_offset_MLP_depth_list, is_logits=True,
                normalization_type=auto_offset_MLP_normalization_type,
                activation_type=auto_offset_MLP_feature_activation_type)
            input_vertex_coordinates = input_vertex_coordinates + offset
        # Gather the destination vertex of the edges
        d_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 1])
        # Prepare initial edge features
        if edgetozero==1:                                                             ######add change
            #d_vertex_coordinates = s_vertex_coordinates
            #s_vertex_features = tf.zeros(tf.shape(s_vertex_features, out_type=tf.dtypes.int32), dtype=tf.dtypes.float32)
            pass
        edge_features = tf.concat(
            [s_vertex_features, s_vertex_coordinates - d_vertex_coordinates],
             axis=-1)
        with tf.variable_scope('extract_vertex_features'):
            # Extract edge features
            edge_features = self._edge_feature_fn(
                edge_features,
                Ks=edge_MLP_depth_list,
                is_logits=False,
                normalization_type=edge_MLP_normalization_type,
                activation_type=edge_MLP_activation_type)
            #print('edge_features = {}'.format(edge_features))
            #print('----------------------')
            #print(edge_features.get_shape().as_list())
            #tf.shape(edge_features, out_type=tf.dtypes.int32, name=None)
            #tf.print(edge_features, output_stream=sys.stderr)
            #print('-----------------------')
            # Aggregate edge features
            #edge_features = 
            #attention(edge_features,edges,t_num_group)
            #attention_v2(edge_features,edges,t_num_group)
            #if edgetozero==1:
            edge_features = attention_v2(edge_features,edges,t_num_group)

            aggregated_edge_features = self._aggregation_fn(
                edge_features,
                edges[:, 1],
                tf.shape(input_vertex_features)[0])
        # Update vertex features
        with tf.variable_scope('combined_features'):
            update_features = self._update_fn(aggregated_edge_features,
                Ks=update_MLP_depth_list, is_logits=True,
                normalization_type=update_MLP_normalization_type,
                activation_type=update_MLP_activation_type)
        output_vertex_features = update_features + input_vertex_features
        return output_vertex_features
