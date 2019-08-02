# coding: utf-8
# some shit I had to do in order to change a 2 to a 1 in my hand-pruned network :P
# (saving this just in case I have to do something similar again)
from asnets import interactive_network as inet
ni = inet.NetworkInstantiator('./sparse/ttw-weights-hand-pruned.pkl', use_lm_cuts=False)
ni.weight_manager
import tensorflow as tf; import numpy as np
sess = tf.Session()
sess.run(tf.global_variables_initializer())
with sess.as_default():
    w_state = ni.weight_manager.__getstate__()
second_act_layer = w_state['act_weights_np'][1]
second_act_layer = {k.schema_name: v for k, v in second_act_layer.items()}
second_act_layer['move_car'][0][0,3,1] = 1
tf.reset_default_graph()
sess.close()
sess = tf.Session()
weights.__setstate__(w_state)
ni.weight_manager.__setstate__(w_state)
sess.run(tf.global_variables_initializer())
with sess.as_default():
    ni.weight_manager.save('./sparse/ttw-weights-hand-pruned-simpler.pkl')
