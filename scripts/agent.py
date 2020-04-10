from model import *
from keras import backend as K
from keras.optimizers import RMSprop

#input for neural networks
###Input for agent_1
a_input_0 = Input(shape=(m_bits,)) #message
a_input_1 = Input(shape=(k_bits,)) #key for agent_1

b_input_1 = Input(shape=(k_bits,)) #key for agent_2
attacker_input = Input(shape=(c_bits,)) #cyphertext


agent_1 = agent_network(m_bits,k_bits,'agent_1')[1]
agent_1_output = agent_1([a_input_0, a_input_1])

agent_2 = agent_network(m_bits,k_bits,'agent_2')[1]
agent_2_output = agent_2([agent_1_output, b_input_1])# agent_2 sees ciphertext AND key

attacker = adv_output(c_bits,k_bits)[1]
attacker_output = attacker(agent_1_output)# attacker doesn't see the key

attacker_loss = K.mean(  K.sum(K.abs(a_input_0 - attacker_output), axis=-1))
agent_2_loss = K.mean(K.sum(K.abs(a_input_0 - agent_2_output), axis=-1)  )
global_loss = agent_2_loss + K.square(m_bits/2 - attacker_loss)/( (m_bits//2)**2 )

global_optim = RMSprop(lr=0.0001)
attacker_optim = RMSprop(lr=0.0001) #default 0.001

#build and compile model for agents interaction.
global_model = Model([a_input_0, a_input_1, b_input_1], agent_2_output, name='global_model')
global_model.add_loss(global_loss)
global_model.compile(optimizer=global_optim)

#build and compile the ATTACKER model. At this stage, agent_1 neural network is frozen.
agent_1.trainable = False
attacker_model = Model([a_input_0, a_input_1], attacker_output, name='attacker_model')
attacker_model.add_loss(attacker_loss)
attacker_model.compile(optimizer=attacker_optim)