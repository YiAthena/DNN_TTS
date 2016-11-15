# pylint:skip-file
import mxnet as mx
import numpy as np
from collections import namedtuple

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_i_weight", "i2h_i_bias",
                                     "h2h_i_weight", "h2h_i_bias",
                                     "i2h_f_weight", "i2h_f_bias",
                                     "h2h_f_weight", "h2h_f_bias",
                                     "i2h_o_weight", "i2h_o_bias",
                                     "h2h_o_weight", "h2h_o_bias",
                                     "i2h_tr_weight", "i2h_tr_bias",
                                     "h2h_tr_weight", "h2h_tr_bias",
                                     
                                     ])
DNNParam = namedtuple("DNNParam", ["fc_weight", "fc_bias"])

def lstm(num_hidden_lstm, indata, prev_state, param, seqidx, layeridx, dropout):
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)

    i2h_i = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_i_weight,
                                bias=param.i2h_i_bias,
                                num_hidden=num_hidden_lstm,
                                name="t%d_l%d_i2h_i" % (seqidx, layeridx))
    h2h_i = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_i_weight,
                                no_bias=True,
                                num_hidden=num_hidden_lstm,
                                name="t%d_l%d_h2h_i" % (seqidx, layeridx))
    gate_i = i2h_i + h2h_i

    i2h_f = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_f_weight,
                                bias=param.i2h_f_bias,
                                num_hidden=num_hidden_lstm,
                                name="t%d_l%d_i2h_f" % (seqidx, layeridx))
    h2h_f = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_f_weight,
                                no_bias=True,
                                num_hidden=num_hidden_lstm,
                                name="t%d_l%d_h2h_f" % (seqidx, layeridx))
    gate_f = i2h_f + h2h_f

    i2h_o = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_o_weight,
                                bias=param.i2h_o_bias,
                                num_hidden=num_hidden_lstm,
                                name="t%d_l%d_i2h_o" % (seqidx, layeridx))
    h2h_o = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_o_weight,
                                no_bias=True,
                                num_hidden=num_hidden_lstm,
                                name="t%d_l%d_h2h_o" % (seqidx, layeridx))
    gate_o = i2h_o + h2h_o

    i2h_tr = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_tr_weight,
                                bias=param.i2h_tr_bias,
                                num_hidden=num_hidden_lstm,
                                name="t%d_l%d_i2h_tr" % (seqidx, layeridx))
    h2h_tr = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_tr_weight,
                                no_bias=True,
                                num_hidden=num_hidden_lstm,
                                name="t%d_l%d_h2h_tr" % (seqidx, layeridx))
    gate_tr = i2h_tr+ h2h_tr

    in_gate = mx.sym.Activation(gate_i, name = "in_g", act_type="sigmoid")
    forget_gate = mx.sym.Activation(gate_f, name = "f_g", act_type="sigmoid")
    out_gate = mx.sym.Activation(gate_o, name = "o_g", act_type="sigmoid")
    in_transform = mx.sym.Activation(gate_tr, name = "tr_g", act_type="tanh")

    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")

    return LSTMState(c=next_c, h=next_h)



def hybrid_net(num_ff_layer, num_hidden_ff, num_lstm_layer, num_hidden_lstm, seq_len, num_label, dropout=0):
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('linear_label')
    
    act_curr = mx.symbol.Variable('act_curr')
    
    dataSlice = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)
    
    #2 DNN layers
    ff_param = []
    out_ff = []
    if num_ff_layer>0:
        for i in range(num_ff_layer):
            ff_param.append(DNNParam(fc_weight=mx.sym.Variable("l%d_fc_weight" % i),
                                    fc_bias=mx.sym.Variable("l%d_fc_bias" % i)
                                    )
            )
        for seqidx in range(seq_len):
            hidden = dataSlice[seqidx]
            for i in range(num_ff_layer):
                if i == 0:
                    fc = mx.symbol.FullyConnected(data = hidden, weight = ff_param[i].fc_weight, bias = ff_param[i].fc_bias, num_hidden=num_hidden_ff, name='fc1')
                else:
                    fc = mx.symbol.FullyConnected(data = act_curr, weight = ff_param[i].fc_weight, bias = ff_param[i].fc_bias, num_hidden=num_hidden_ff, name='fc1')
                
                act_curr = mx.symbol.Activation(data = fc, name='tanh1', act_type="tanh")
                        
            out_ff.append(act_curr)
    else:
        out_ff.append(dataSlice)
   # fc1 = mx.symbol.FullyConnected(data = data, weight = fc1_weight, bias = fc1_bias, num_hidden=num_hidden_ff, name='fc1')
  #  act1 = mx.symbol.Activation(data = fc1, name='tanh1', act_type="tanh")

  #  fc2 = mx.symbol.FullyConnected(data = data, weight = fc2_weight, bias = fc2_bias, num_hidden=num_hidden_ff, name='fc2')
  #  act2 = mx.symbol.Activation(data = fc2, name='tanh2', act_type="tanh")
    
    
    #2 LSTM layers
    param_cells = []
    last_states = []


    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_i_weight=mx.sym.Variable("l%d_i2h_i_weight" % i),
                                     i2h_i_bias=mx.sym.Variable("l%d_i2h_i_bias" % i),
                                     h2h_i_weight=mx.sym.Variable("l%d_h2h_i_weight" % i),
                                     h2h_i_bias=mx.sym.Variable("l%d_h2h_i_bias" % i),

                                     i2h_f_weight=mx.sym.Variable("l%d_i2h_f_weight" % i),
                                     i2h_f_bias=mx.sym.Variable("l%d_i2h_f_bias" % i),
                                     h2h_f_weight=mx.sym.Variable("l%d_h2h_f_weight" % i),
                                     h2h_f_bias=mx.sym.Variable("l%d_h2h_f_bias" % i),

                                     i2h_o_weight=mx.sym.Variable("l%d_i2h_o_weight" % i),
                                     i2h_o_bias=mx.sym.Variable("l%d_i2h_o_bias" % i),
                                     h2h_o_weight=mx.sym.Variable("l%d_h2h_o_weight" % i),
                                     h2h_o_bias=mx.sym.Variable("l%d_h2h_o_bias" % i),

                                     i2h_tr_weight=mx.sym.Variable("l%d_i2h_tr_weight" % i),
                                     i2h_tr_bias=mx.sym.Variable("l%d_i2h_tr_bias" % i),
                                     h2h_tr_weight=mx.sym.Variable("l%d_h2h_tr_weight" % i),
                                     h2h_tr_bias=mx.sym.Variable("l%d_h2h_tr_bias" % i)

                                     )
        )
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

# dataSlice = mx.sym.SliceChannel(data=out_ff, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []

    for seqidx in range(seq_len):
        hidden = out_ff[seqidx]
        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp = 0.
            else:
                dp = dropout
            
            next_state = lstm(num_hidden_lstm=num_hidden_lstm, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp)
            hidden = next_state.h
            last_states[i] = next_state

        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=1)

    hidden_final = mx.sym.Reshape(hidden_concat, target_shape=(0, num_hidden_lstm))
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
#num_label is the dim of label
    pred = mx.sym.FullyConnected(data=hidden_final, num_hidden=num_label,
                                 weight=cls_weight, bias=cls_bias, name='pred')
    pred = mx.sym.Reshape(pred, target_shape=(0, seq_len, num_label))

    sm = mx.sym.LinearRegressionOutput(data=pred, label=label, name='linear_label')

    return sm

def lstm_unroll(num_lstm_layer, seq_len, num_hidden_lstm, num_label, dropout):
    param_cells = []
    last_states = []
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('linear_label')
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")

    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_i_weight=mx.sym.Variable("l%d_i2h_i_weight" % i),
                                     i2h_i_bias=mx.sym.Variable("l%d_i2h_i_bias" % i),
                                     h2h_i_weight=mx.sym.Variable("l%d_h2h_i_weight" % i),
                                     h2h_i_bias=mx.sym.Variable("l%d_h2h_i_bias" % i),

                                     i2h_f_weight=mx.sym.Variable("l%d_i2h_f_weight" % i),
                                     i2h_f_bias=mx.sym.Variable("l%d_i2h_f_bias" % i),
                                     h2h_f_weight=mx.sym.Variable("l%d_h2h_f_weight" % i),
                                     h2h_f_bias=mx.sym.Variable("l%d_h2h_f_bias" % i),

                                     i2h_o_weight=mx.sym.Variable("l%d_i2h_o_weight" % i),
                                     i2h_o_bias=mx.sym.Variable("l%d_i2h_o_bias" % i),
                                     h2h_o_weight=mx.sym.Variable("l%d_h2h_o_weight" % i),
                                     h2h_o_bias=mx.sym.Variable("l%d_h2h_o_bias" % i),

                                     i2h_tr_weight=mx.sym.Variable("l%d_i2h_tr_weight" % i),
                                     i2h_tr_bias=mx.sym.Variable("l%d_i2h_tr_bias" % i),
                                     h2h_tr_weight=mx.sym.Variable("l%d_h2h_tr_weight" % i),
                                     h2h_tr_bias=mx.sym.Variable("l%d_h2h_tr_bias" % i)

                                     )
        )
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)
    dataSlice = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = dataSlice[seqidx]

        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp = 0.
            else:
                dp = dropout
      
            next_state = lstm(num_hidden_lstm=num_hidden_lstm, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=1)
    hidden_final = mx.sym.Reshape(hidden_concat, target_shape=(0, num_hidden_lstm))
    pred = mx.sym.FullyConnected(data=hidden_final, num_hidden=num_label,
                                 weight=cls_weight, bias=cls_bias, name='pred')
    pred = mx.sym.Reshape(pred, target_shape=(0, seq_len, num_label))
    sm = mx.sym.LinearRegressionOutput(data=pred, label=label, name='linear_label')

    return sm


