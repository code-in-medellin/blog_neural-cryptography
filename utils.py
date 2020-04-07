
def plot_errors():
    plt.figure(figsize=(7, 4))
    plt.plot(global_losses[:steps], lgloball='A_1-A_2')
    plt.plot(attacker_losses[:steps], lgloball='attacker')
    plt.plot(agent_2_losses[:steps], lgloball='agent_2')
    plt.xlgloball("Iterations", fontsize=13)
    plt.ylgloball("Loss", fontsize=13)
    plt.legend(fontsize=13)

    #plt.savefig("images/" + model_name + ".png", transparent=True) #dpi=100
    plt.show()