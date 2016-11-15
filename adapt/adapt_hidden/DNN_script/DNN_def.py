import mxnet as mx
import numpy as np
from collections import namedtuple
import logging
logging.basicConfig(level=logging.DEBUG)



def build_network():
    data = mx.symbol.Variable('data')
    Y = mx.symbol.Variable('softmax_label')

    fc1_weight = mx.symbol.Variable('fc1_weight')
    fc2_weight = mx.symbol.Variable('fc2_weight')
    fc3_weight = mx.symbol.Variable('fc3_weight')
    fc4_weight = mx.symbol.Variable('fc4_weight')

    fc1_bias = mx.symbol.Variable('fc1_bias')
    fc2_bias = mx.symbol.Variable('fc2_bias')
    fc3_bias = mx.symbol.Variable('fc3_bias')
    fc4_bias = mx.symbol.Variable('fc4_bias')


    fc1 = mx.symbol.FullyConnected(data = data,  weight = fc1_weight, bias = fc1_bias, name='fc1', num_hidden=512)
   # sc1 = mx.symbol.FullyConnected(data = sc1 ,  weight = sc1_weight, no_bias=True, name='sc1', num_hidden=512)
   # fc1 = fc1 + sc1
    act1 = mx.symbol.Activation(data = fc1, name='tanh1', act_type="tanh")

    fc2 = mx.symbol.FullyConnected(data = act1, weight = fc2_weight, bias = fc2_bias, name = 'fc2', num_hidden = 512)
   # sc2 = mx.symbol.FullyConnected(data = sc2 ,  weight = sc2_weight, no_bias=True, name='sc1', num_hidden=512)
   # fc2 = fc2 + sc2
    act2 = mx.symbol.Activation(data = fc2, name='tanh2', act_type="tanh")

    fc3 = mx.symbol.FullyConnected(data = act2, weight = fc3_weight, bias = fc3_bias, name = 'fc3', num_hidden = 512)
  #  sc3= mx.symbol.FullyConnected(data = sc3 ,  weight = sc3_weight, no_bias=True, name='sc1', num_hidden=512)
  #  fc3 = fc3+ sc3
    act3 = mx.symbol.Activation(data = fc3, name='tanh3', act_type="tanh")

    fc4 = mx.symbol.FullyConnected(data = act3, weight = fc4_weight, bias = fc4_bias, name = 'fc4', num_hidden = 512)
  #  sc3= mx.symbol.FullyConnected(data = sc3 ,  weight = sc3_weight, no_bias=True, name='sc1', num_hidden=512)
  #  fc3 = fc3+ sc3
    act4 = mx.symbol.Activation(data = fc4, name='tanh4', act_type="tanh")

    fo = mx.symbol.FullyConnected(data = act4, name = 'fo', num_hidden=202)
    sm = mx.symbol.LinearRegressionOutput(data = fo, label = Y, name = 'softmax_label')

    return sm

def adapt_network():
    data = mx.symbol.Variable('data')
    sc = mx.symbol.Variable('sc')
    Y = mx.symbol.Variable('softmax_label')

    fc1_weight = mx.symbol.Variable('fc1_weight')
    fc2_weight = mx.symbol.Variable('fc2_weight')
    fc3_weight = mx.symbol.Variable('fc3_weight')
    fc4_weight = mx.symbol.Variable('fc4_weight')

    fc1_bias = mx.symbol.Variable('fc1_bias')
    fc2_bias = mx.symbol.Variable('fc2_bias')
    fc3_bias = mx.symbol.Variable('fc3_bias')
    fc4_bias = mx.symbol.Variable('fc4_bias')

    sc1_weight = mx.symbol.Variable('sc1_weight')
    sc2_weight = mx.symbol.Variable('sc2_weight')
    sc3_weight = mx.symbol.Variable('sc3_weight')
    sc4_weight = mx.symbol.Variable('sc4_weight')


    fc1 = mx.symbol.FullyConnected(data = data,  weight = fc1_weight, bias = fc1_bias, name='fc1', num_hidden=512)
    # sc1 = mx.symbol.FullyConnected(data = sc,  weight = sc1_weight, no_bias=True, name='sc1', num_hidden=512)
    # fc1 = fc1 + sc1
    act1 = mx.symbol.Activation(data = fc1, name='tanh1', act_type="tanh")

    fc2 = mx.symbol.FullyConnected(data = act1, weight = fc2_weight, bias = fc2_bias, name = 'fc2', num_hidden = 512)
    sc2 = mx.symbol.FullyConnected(data = sc ,  weight = sc2_weight, no_bias=True, name='sc2', num_hidden=512)
    fc2 = fc2 + sc2
    act2 = mx.symbol.Activation(data = fc2, name='tanh2', act_type="tanh")

    fc3 = mx.symbol.FullyConnected(data = act2, weight = fc3_weight, bias = fc3_bias, name = 'fc3', num_hidden = 512)
    # sc3= mx.symbol.FullyConnected(data = sc ,  weight = sc3_weight, no_bias=True, name='sc3', num_hidden=512)
    # fc3 = fc3+ sc3
    act3 = mx.symbol.Activation(data = fc3, name='tanh3', act_type="tanh")

    fc4 = mx.symbol.FullyConnected(data = act3, weight = fc4_weight, bias = fc4_bias, name = 'fc4', num_hidden = 512)
    # sc4= mx.symbol.FullyConnected(data = sc ,  weight = sc4_weight, no_bias=True, name='sc4', num_hidden=512)
    # fc4 = fc4+ sc4
    act4 = mx.symbol.Activation(data = fc4, name='tanh4', act_type="tanh")

    fo = mx.symbol.FullyConnected(data = act4, name = 'fo', num_hidden=202)
    sm = mx.symbol.LinearRegressionOutput(data = fo, label = Y, name = 'softmax_label')

    return sm

def sc_network():
    data = mx.symbol.Variable('data')
    Y = mx.symbol.Variable('softmax_label')

    fc1_weight = mx.symbol.Variable('fc1_weight')
    fc2_weight = mx.symbol.Variable('fc2_weight')
    fc3_weight = mx.symbol.Variable('fc3_weight')
    fc4_weight = mx.symbol.Variable('fc4_weight')

    B1_bias = mx.symbol.Variable('B1_bias')
    B2_bias = mx.symbol.Variable('B2_bias')
    B3_bias = mx.symbol.Variable('B3_bias')
    B4_bias = mx.symbol.Variable('B4_bias')

    fc1_bias = mx.symbol.Variable('fc1_bias')
    fc2_bias = mx.symbol.Variable('fc2_bias')
    fc3_bias = mx.symbol.Variable('fc3_bias')
    fc4_bias = mx.symbol.Variable('fc4_bias')


    fc1 = mx.symbol.FullyConnected(data = data,  weight = fc1_weight, bias = B1_bias, name='fc1', num_hidden=512)
   # sc1 = mx.symbol.FullyConnected(data = sc1 ,  weight = sc1_weight, no_bias=True, name='sc1', num_hidden=512)
   # fc1 = fc1 + sc1
    act1 = mx.symbol.Activation(data = fc1, name='tanh1', act_type="tanh")

    fc2 = mx.symbol.FullyConnected(data = act1, weight = fc2_weight, bias = B2_bias, name = 'fc2', num_hidden = 512)
   # sc2 = mx.symbol.FullyConnected(data = sc2 ,  weight = sc2_weight, no_bias=True, name='sc1', num_hidden=512)
   # fc2 = fc2 + sc2
    act2 = mx.symbol.Activation(data = fc2, name='tanh2', act_type="tanh")

    fc3 = mx.symbol.FullyConnected(data = act2, weight = fc3_weight, bias = B3_bias, name = 'fc3', num_hidden = 512)
  #  sc3= mx.symbol.FullyConnected(data = sc3 ,  weight = sc3_weight, no_bias=True, name='sc1', num_hidden=512)
  #  fc3 = fc3+ sc3
    act3 = mx.symbol.Activation(data = fc3, name='tanh3', act_type="tanh")

    fc4 = mx.symbol.FullyConnected(data = act3, weight = fc4_weight, bias = B4_bias, name = 'fc4', num_hidden = 512)
  #  sc3= mx.symbol.FullyConnected(data = sc3 ,  weight = sc3_weight, no_bias=True, name='sc1', num_hidden=512)
  #  fc3 = fc3+ sc3
    act4 = mx.symbol.Activation(data = fc4, name='tanh4', act_type="tanh")

    fo = mx.symbol.FullyConnected(data = act4, name = 'fo', num_hidden=202)
    sm = mx.symbol.LinearRegressionOutput(data = fo, label = Y, name = 'softmax_label')

    return sm