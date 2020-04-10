from keras.models import Model
from keras.engine.input_layer import Input
from keras.layers.core import Activation, Dense
from keras.layers import Flatten, Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.merge import concatenate

#Problem parameters: message, key, and ciphertext bit lengths
m_bits = 16
k_bits = 16
c_bits = 16

pad = 'same' #for padding parameter in Keras

# Compute the size of the message space, used later in training
m_train = 2**(m_bits) #+ k_bits)


def agent_network(m_bits, k_bits, name):
    a_input_0 = Input(shape=(m_bits,))  # message
    a_input_1 = Input(shape=(k_bits,))  # key for agent_1
    # we use functional writing
    a_input = concatenate([a_input_0, a_input_1], axis=1)

    ###Neural Network architecture
    a_dense1 = Dense(units=(m_bits + k_bits))(a_input)
    a_dense1a = Activation('tanh')(a_dense1)
    a_reshape = Reshape((m_bits + k_bits, 1,))(a_dense1a)

    a_conv1 = Conv1D(filters=2, kernel_size=4, strides=1, padding=pad)(a_reshape)
    a_conv1a = Activation('tanh')(a_conv1)

    a_conv2 = Conv1D(filters=4, kernel_size=4, strides=2, padding=pad)(a_conv1a)
    a_conv2a = Activation('tanh')(a_conv2)

    a_conv3 = Conv1D(filters=4, kernel_size=1, strides=1, padding=pad)(a_conv2a)
    a_conv3a = Activation('tanh')(a_conv3)

    a_conv4 = Conv1D(filters=1, kernel_size=1, strides=1, padding=pad)(a_conv3a)
    a_conv4a = Activation('sigmoid')(a_conv4)

    a_output = Flatten()(a_conv4a)

    model = Model([a_input_0, a_input_1], a_output, name=name)
    return a_output, model


def adv_output(c_bits, k_bits):
    einput = Input(shape=(c_bits,))  # ciphertext only
    e_dense1 = Dense(units=(c_bits + k_bits))(einput)
    e_dense1a = Activation('tanh')(e_dense1)
    e_dense2 = Dense(units=(c_bits + k_bits))(e_dense1a)
    e_dense2a = Activation('tanh')(e_dense2)
    e_reshape = Reshape((c_bits + k_bits, 1,))(e_dense2a)

    e_conv1 = Conv1D(filters=2, kernel_size=4, strides=1, padding=pad)(e_reshape)
    e_conv1a = Activation('tanh')(e_conv1)

    e_conv2 = Conv1D(filters=4, kernel_size=2, strides=2, padding=pad)(e_conv1a)
    e_conv2a = Activation('tanh')(e_conv2)

    e_conv3 = Conv1D(filters=4, kernel_size=1, strides=1, padding=pad)(e_conv2a)
    e_conv3a = Activation('tanh')(e_conv3)

    e_conv4 = Conv1D(filters=1, kernel_size=1, strides=1, padding=pad)(e_conv3a)
    e_conv4a = Activation('sigmoid')(e_conv4)

    # adversary attempt at guessing the plaintext
    attacker_output = Flatten()(e_conv4a)
    attacker_model = Model(einput, attacker_output, name='attacker')
    return attacker_output, attacker_model