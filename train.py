import numpy as np
import sys

from agent import *

global_losses = []
agent_2_losses = []
attacker_losses = []

n_epochs = 20
batch_size = 512
n_batches = m_train // batch_size


global_cycles = 1
attacker_cycles = 2

epoch = 0
print("Training for", n_epochs, "epochs with", n_batches, "batches of size", batch_size)

while epoch < n_epochs:
    global_losses_0 = []
    agent_2_losses_0 = []
    attacker_losses_0 = []
    for iteration in range(n_batches):

        # Train the A-B+E network
        #
        agent_1.trainable = True
        for cycle in range(global_cycles):
            # Select a random batch of messages, and a random batch of keys
            #
            m_batch = np.random.randint(0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
            k_batch = np.random.randint(0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
            loss = global_model.train_on_batch([m_batch, k_batch, k_batch], None)

        global_losses_0.append(loss)
        global_losses.append(loss)
        global_avg = np.mean(global_losses_0)

        # Evaluate agent_2's ability to decrypt a message
        m_enc = agent_1.predict([m_batch, k_batch])
        m_dec = agent_2.predict([m_enc, k_batch])
        loss = np.mean(np.sum(np.abs(m_batch - m_dec), axis=-1))
        agent_2_losses_0.append(loss)
        agent_2_losses.append(loss)
        agent_2_avg = np.mean(agent_2_losses_0)

        # Train the attacker network
        #
        agent_1.trainable = False
        for cycle in range(attacker_cycles):
            m_batch = np.random.randint(0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
            k_batch = np.random.randint(0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
            loss = attacker_model.train_on_batch([m_batch, k_batch], None)

        attacker_losses_0.append(loss)
        attacker_losses.append(loss)
        attacker_avg = np.mean(attacker_losses_0)

        if iteration % max(1, (n_batches // 100)) == 0:
            print("\rEpoch {:3}: {:3}% | global: {:2.3f} | attacker: {:2.3f} | agent_2: {:2.3f}".format(
                epoch, 100 * iteration // n_batches, global_avg, attacker_avg, agent_2_avg), end="")
            sys.stdout.flush()

    print()
    epoch += 1

print('Training finished.')