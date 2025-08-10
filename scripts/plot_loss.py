import matplotlib.pyplot as plt

# Function to read log file and extract epoch numbers and loss values
def read_log_file(log_file):
    epochs = []
    losses = []
    
    with open(log_file, 'r') as file:
        for line in file:
            epoch, loss = line.strip().split()
            epochs.append(int(epoch))
            losses.append(float(loss))
    
    return epochs, losses

# Function to plot the loss with respect to the epoch
def plot_loss(epochs, losses):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.grid(True)
    plt.show()

# Main function
if __name__ == "__main__":
    log_file = 'loss.log'  # Replace with the path to your log file
    epochs, losses = read_log_file(log_file)
    plot_loss(epochs, losses)
