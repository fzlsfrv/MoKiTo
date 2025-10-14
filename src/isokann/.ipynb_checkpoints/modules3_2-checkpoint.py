import numpy as np
import torch as pt
import scipy
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import logging
import sys

# For reproducibility
np.random.seed(0)
pt.manual_seed(0)
random.seed(0)

# Check if CUDA is available, otherwise use CPU
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

def scale_and_shift(y):

    minarr = pt.min(y)
    maxarr = pt.max(y)
    hat_y  = (y - minarr) / (maxarr - minarr)

    return hat_y
    
class NeuralNetwork(pt.nn.Module):
    def __init__(self, Nodes, enforce_positive=0, activation_function = 'sigmoid', LeakyReLU_par = 0.01):
        super(NeuralNetwork, self).__init__()

        # self parameters
        self.input_size        = Nodes[0]
        self.output_size       = Nodes[-1]
        self.NhiddenLayers     = len(Nodes) - 2
        self.Nodes             = Nodes
        
        self.enforce_positive  = enforce_positive

        # build NN architecture
        self.hidden_layers = pt.nn.ModuleList()

        # add layers
        self.hidden_layers.extend([pt.nn.Linear(self.input_size,    self.Nodes[1])])
        self.hidden_layers.extend([pt.nn.Linear(self.Nodes[1+l], self.Nodes[1+l+1]) for l in range(self.NhiddenLayers)])

        # the output of the last layer must be equal to 1
        #if self.Nodes[-1] > 1:
        #    self.hidden_layers.extend([pt.nn.Linear(self.Nodes[-1], 1)])

        # define activation function

        if activation_function == 'sigmoid':
            self.activation  = pt.nn.Sigmoid()  # #
        elif activation_function == 'relu':
            self.activation  = pt.nn.ReLU()
        elif activation_function == 'leakyrelu': 
            self.activation  = pt.nn.LeakyReLU(LeakyReLU_par)
        
        self.activation2  = pt.nn.Softplus(10)
        
        
    def forward(self, X):

        # Pass input through each hidden layer but the last one
        for layer in self.hidden_layers[:-1]:
            X = self.activation(layer(X))

        # Apply the last layer (but not the activation function)
        X = self.hidden_layers[-1](X)
        
        if self.enforce_positive == 1:
            X= self.activation2(X)  #.unsqueeze(1)

        return X.squeeze()

def trainNN(net,
            lr,
            wd,
            Nepochs,
            batch_size,
            momentum,
            patience,
            X,
            Y,
            test_size = 0.2
           ):

    best_loss = float('inf')
    best_model = None
    patience_counter = 0

    # Split training and validation data
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=42)

    # Define the optimizer
    optimizer = pt.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum = momentum, nesterov=True)
    #optimizer = pt.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    #optimizer = pt.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

    # Define the loss function
    MSE = pt.nn.MSELoss()

    # Define an array where to store the loss
    train_losses = []
    val_losses = []
    
    # Train the model
    for epoch in range(Nepochs):

        permutation = pt.randperm(X_train.size()[0], device=device)

        for i in range(0, X_train.size()[0], batch_size):

            # Clear gradients for next training
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]

            batch_x, batch_y = X_train[indices], Y_train[indices]

            # Make a new prediction
            new_points  =  net( batch_x )

            # measure the MSE
            loss = MSE(batch_y, new_points)

            # computes the gradients of the loss with respect to the model parameters using backpropagation.
            loss.backward()

            # updates the NN parameters
            optimizer.step()

        train_losses.append(loss.item())

        # Validation
        with pt.no_grad():
            val_outputs = net(X_val)
            val_loss    = MSE(val_outputs, Y_val)
            val_losses.append(val_loss.item())

        # Early stopping

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = net.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
            if patience_counter >= patience:
                #print(f"Early stopping at epoch {epoch+1}")
                break

        #print(f'Epoch {epoch+1}/{Nepochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

    return train_losses, val_losses, best_loss


# Random search for hyperparameters optimization:
def random_search(
                    X,
                    Y,
                    NN_epochs,
                    NN_nodes,
                    NN_lr,
                    NN_wd,
                    NN_bs,
                    NN_mu,
                    NN_patience,
                    NN_act_fun,
                    search_iterations=20,
                    test_size = 0.2,
                    out_dir = 'output/isokann/'
                ):

    best_hyperparams  = None
    best_val_loss     = float('inf')
    best_convergence  = float('inf')

    for _ in tqdm(range(search_iterations)):

        Nepochs    = random.choice(NN_epochs)
        nodes      = np.asarray(random.choice(NN_nodes))
        lr         = random.choice(NN_lr)
        wd         = random.choice(NN_wd)
        batch_size = random.choice(NN_bs)
        momentum = random.choice(NN_mu)
        patience   = random.choice(NN_patience)
        act_fun    = random.choice(NN_act_fun)

        print(" ")
        print("Nepochs =",             Nepochs)
        print("Nodes =",               nodes)
        print("Learning rate =",       lr)
        print("Weight decay =",        wd)
        print("Batch size =",          batch_size)
        print("Momentum =",          momentum)
        print("Patience =",            patience)
        print("Activation function =", act_fun)

        f_NN = NeuralNetwork( Nodes = nodes, activation_function = act_fun ).to(device)

        train_losses, val_losses, val_loss, convergence = power_method(X, Y,
                                                          f_NN,
                                                          scale_and_shift,
                                                          Niters = 100,
                                                          Nepochs = Nepochs,
                                                          tolerance  = 1e-3,
                                                          lr = lr,
                                                          wd = wd,
                                                          batch_size = batch_size,
                                                          momentum = momentum,
                                                          patience = patience,
                                                          test_size = test_size)

        print("Validation loss:", val_loss)
        print("Convergence:", convergence[-1])

        if val_loss < best_val_loss and abs(convergence[-1] - 1) < 0.05:
            best_val_loss = val_loss

            best_hyperparams = {'Nepochs'        : Nepochs,
                                'nodes'          : nodes,
                                'learning_rate'  : lr,
                                'weight_decay'   : wd,
                                'batch_size'     : batch_size,
                                'momentum'       : momentum,
                                'patience'       : patience,
                                'act_fun'        : act_fun}

        logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(out_dir + "logs/isokann_rs_job_output_1.txt"), logging.StreamHandler(sys.stdout)],
                    force=True,  
                )

        logging.info(f"Nepochs={Nepochs} nodes={nodes} lr={lr} wd={wd} bs={batch_size} patience={patience} act={act_fun}")
        logging.info(f"Validation loss: {val_loss:.6f}  Convergence: {convergence[-1]:.4f}")

        del f_NN

    return best_hyperparams, best_val_loss


def power_method(   pt_x0,
                    pt_xt,
                    f_NN,
                    scale_and_shift,
                    Niters = 500,
                    Nepochs = 10,
                    lr = 1e-3,
                    wd = 0,
                    batch_size = 50,
                    momentum = 0.9,
                    patience = 10,
                    tolerance  = 5e-3,
                    test_size = 0.2,
                    print_eta=False,
                    loss = 'iterations'
                ):

    """
    train_LOSS, val_LOSS, best_loss = power_method(pt_x0, pt_y, f_NN, scale_and_shift, Niters = 500, tolerance  = 5e-3)
    """
    train_LOSS = np.empty(0, dtype = object)
    val_LOSS   = np.empty(0, dtype = object)
    convergence = []

    if   print_eta == False:
        loop = range(Niters)
    elif print_eta == True:
        loop = tqdm(range(Niters))

    for i in loop:

        old_chi =  f_NN(pt_x0).cpu().detach().numpy()
        ##if i==1:
            #logging.basicConfig(
                    #level=logging.INFO,
                    #format="%(asctime)s %(message)s",
                    #force=True,  
                #)
           # logging.info(f"old_chi: {old_chi[0:10]}")

        pt_chi  =  f_NN( pt_xt )

        if pt_chi.dim() == 1:
            pt_chi = pt_chi.unsqueeze(1)

        pt_y    =  pt.mean(pt_chi, axis=1)
        pt_y       =  scale_and_shift(pt_y).to(device)
        pt_y                            =  pt_y.clone().detach().requires_grad_(False)  
        
        train_loss, val_loss, best_loss = trainNN(net      = f_NN,
                                                  lr       = lr,
                                                  wd       = wd,
                                                  Nepochs    = Nepochs,
                                                  batch_size = batch_size,
                                                  momentum = momentum,
                                                  patience   = patience,
                                                  test_size  = test_size,
                                                  X          = pt_x0,
                                                  Y          = pt_y)


        if loss == 'iterations':
            train_LOSS           = np.append(train_LOSS, train_loss[-1])
            val_LOSS             = np.append(val_LOSS, val_loss[-1])
        elif loss == 'full':
            train_LOSS           = np.append(train_LOSS, train_loss)
            val_LOSS             = np.append(val_LOSS, val_loss)

        new_chi   = f_NN(pt_x0).cpu().detach().numpy()

        slope = scipy.stats.linregress(old_chi, new_chi).slope
        convergence.append( slope )

        if slope < 1 + tolerance and slope > 1 - tolerance:
            break

    return train_LOSS, val_LOSS, best_loss, convergence

def exit_rates_from_chi(tau, chi_0, chi_tau):
    
    #
    chi1      = chi_0
    chi2      = 1- chi_0

    #
    prop_chi1 = chi_tau
    prop_chi2 = 1 - chi_tau

    res1 = scipy.stats.linregress(chi1, prop_chi1)
    res2 = scipy.stats.linregress(chi2, prop_chi2)

    rate1  = - 1 / tau * np.log(  np.abs(res1.slope)  ) * ( 1 + res1.intercept  / ( res1.slope - 1 ))
    rate2  = - 1 / tau * np.log(  np.abs(res2.slope) ) * ( 1 + res2.intercept  / ( res2.slope - 1 ))
    
    Qc = np.array([[-rate1, rate1], [rate2, -rate2]])
    #
    print(r"Slope $\chi$:", res1.slope)
    print(r"Intercept $\chi$:", res1.intercept)
    print(" ")
    print(r"Slope $1-\chi$:", res2.slope)
    print(r"Intercept $1-\chi$:", res2.intercept)
    print(" ")
    print('Exit rate 1:', rate1)
    print('Exit rate 2:', rate2)
    print(" ")
    #
    print('Rate matrix:')
    print(Qc)

    return rate1, rate2, Qc
