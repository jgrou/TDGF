import torch
import numpy as np
import time
import DGM_torch as DGM
import DNM_torch_xp as DNM
import COS_method as cos
torch.manual_seed(12)

# Option parameters
T = 1.0
r = 0.05            # Interest rate
option = 'call'     # Type: 'call'/'put'
name = 'BS_'+option # File name of everything that we save

if option == 'call':
    CP = 1
    sigma = 0.25
elif option == 'put':
    CP = -1
    sigma = 0.5 # Higher volatility to observe difference American and European

# Terminal pay-off option
def Phi(x):
    return torch.maximum(torch.tensor(0.0).to('cuda'), CP * (x - 1.0))

# Neural network parameters
nodes_per_layer = 50
d = 1 # Number of parameters that enter the DGM network: S

# Sampling parameters
nSim_dim = 600         # Number of simulations per dimension
nSim_DGM = nSim_dim * (d+1)              
nSim_TDGF = nSim_dim * d      
S_low = 0.01           
S_high = 3.0
Omega = S_high - S_low # Domain size

# TDGF parameters
N_t = 100
h = T / N_t

# Training parameters
sampling_stages_DNM = 2000                      # Number of times train TDGF
sampling_stages_DGM = N_t * sampling_stages_DNM # Number of times train DGM
        
# Sampling function - randomly sample time-space pairs 
def sampler(nSim):  
    t = T * torch.rand([nSim,1])
    S = S_low + torch.rand([nSim,1]) * Omega
    return t.requires_grad_().to('cuda'), S.requires_grad_().to('cuda')

# Loss function for DGM
def lossDGM(t_int, S_int, S_term, t_bound, S_bound):
    # The option pricing PDE 
    f = DGM_model(t_int, S_int)
    f_t = torch.autograd.grad(f, t_int, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    f_S = torch.autograd.grad(f, S_int, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    f_SS = torch.autograd.grad(f_S, S_int, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    PDE = -f_t + 0.5 * sigma**2 * S_int**2 * f_SS + r * S_int * f_S - r * f
    L1 = torch.mean(PDE**2)

    # Boundary condition
    f_bound = DGM_model(t_bound, S_bound)
    bound_S = torch.autograd.grad(f_bound, S_bound, grad_outputs=torch.ones_like(f_bound), create_graph=True)[0]
    L3 = 0.5 * (CP + 1) * torch.mean((bound_S - torch.tensor(1.0).to('cuda'))**2)

    L2 = torch.mean((DGM_model(t_terminal, S_term) - Phi(S_term))**2) # Loss terminal condition
    return T * Omega * L1 + Omega * L2 + T * L3

# Loss function for TDGF
def lossDNM(S_int):
    # Lagrangian corresponding to the symmetric part of the PDE
    f = DNM_model(S_int)
    f_S = torch.autograd.grad(f, S_int, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    a = 0.5 * sigma**2 * S_int**2
    Lagrangian = 0.5 * (a * f_S**2 + r * f**2)
    G1 = torch.mean(Lagrangian)
    
    # Non-symmetric part of the PDE
    f_old = old_model(S_int)
    old_S = torch.autograd.grad(f_old, S_int, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    b = (sigma**2 - r) * S_int
    F = b * old_S
    G2 = torch.mean(F * f)
    
    G3 = torch.mean((f - f_old)**2)
    return 0.5 * G3 + h * (G1 + G2)

# Loss function for TDGF
def lossTerminal(S_int):
    return torch.mean((DNM_model(S_int) - Phi(S_int))**2)

# Set up network 
DGM_model = DGM.DGMNet(nodes_per_layer, d+1).to('cuda')
DNM_model = DNM.DGMNet(nodes_per_layer, d, r=r, t=0.001, CP=CP).to('cuda')
old_model = DNM.DGMNet(nodes_per_layer, d, r=r, t=0.0, CP=CP).to('cuda')
t_terminal = torch.zeros((nSim_TDGF, 1)).to('cuda')

# Custom learning rate scheduler
def lr_DGM(epoch):
    if epoch <= 10_000:
        return 1e-2
    elif epoch <= 40_000:
        return 1e-3
    else:
        return 1e-4

# Train DGM network
start_time = time.time()
optimizerDGM = torch.optim.Adam(DGM_model.parameters(), 1e-2)
schedulerDGM = torch.optim.lr_scheduler.LambdaLR(optimizerDGM, lr_DGM)

for j in range(sampling_stages_DGM):
    # Generate random points for training
    t_train, S_train = sampler(nSim_DGM)
    t_boundary, S_terminal = sampler(nSim_TDGF)
    S_boundary = S_high * torch.ones((nSim_TDGF, 1)).requires_grad_().to('cuda') # grad resets after backward
    
    optimizerDGM.zero_grad()
    loss = lossDGM(t_train, S_train, S_terminal, t_boundary, S_boundary)
    loss.backward()
    optimizerDGM.step()
            
# Save DGM model
filename = 'weights/'+name
torch.save(DGM_model, filename)
training_time_DGM = time.time() - start_time
print('Training time DGM:', training_time_DGM)

# Train TDGF network for each time
start_time = time.time()
optimizer = torch.optim.Adam(DNM_model.parameters(), 3e-4)    
for j in range(sampling_stages_DNM):                
    _, S_train = sampler(nSim_TDGF)
    optimizer.zero_grad()
    loss = lossTerminal(S_train) 
    loss.backward()
    optimizer.step()
        
# Save initial model
filename = 'weights/'+name+'_t=0.0'
torch.save(DNM_model, filename)

for curr_t in torch.linspace(h, T, N_t): 
    DNM_model.t = curr_t
    optimizer = torch.optim.Adam(DNM_model.parameters(), 3e-4)    
    for j in range(sampling_stages_DNM):                
        _, S_train = sampler(nSim_TDGF)
        optimizer.zero_grad()
        loss = lossDNM(S_train) 
        loss.backward()
        optimizer.step()
            
    # Save/load TDGF model
    filename = 'weights/'+name+'_t='+str(round(float(curr_t),2))
    torch.save(DNM_model, filename)
    old_model.t = curr_t
    old_model = torch.load(filename)
training_time_TDGF = time.time() - start_time
print('Training time TDGF:', training_time_TDGF)

# Plot parameters
n_plot = 47                                     # Points on plot grid
nSamples = 34                                   # Number of time points
S_plot = torch.reshape(torch.linspace(S_low, S_high, n_plot), [-1,1]).to('cuda')
X_plot = np.array(S_plot.cpu())
times = np.linspace(h, T, nSamples)             # Numpy linspace necessary for cos method
title = ["t={:.2f}".format(t) for t in times]
labels = ['Exact', 'DGM', 'TDGF']
optionValue = np.zeros((3, nSamples, n_plot))   # Matrix of option prices for different maturities and spot prices
BS = cos.Black_Scholes(r=r, sigma=sigma, CP=CP) # Define class 

# Evaluate exact option price at each time
start_time = time.time()
for i, curr_t in enumerate(times):
    optionValue[0,i] = np.transpose(BS.OptionPrice(X_plot, curr_t))
computing_time = (time.time() - start_time) / nSamples
print('Computing time Exact:', computing_time)

# Evaluate DGM option price at each time
start_time = time.time()
DGM_model = torch.load('weights/'+name)
for i, curr_t in enumerate(times):
    t_plot = curr_t * torch.ones(n_plot,1).to('cuda')
    optionValue[1,i] = np.transpose(DGM_model(t_plot, S_plot).cpu().detach())
computing_time = (time.time() - start_time) / nSamples
print('Computing time DGM:', computing_time)

# Evaluate TDGF option price at each time
start_time = time.time()
for i, curr_t in enumerate(times):
    DNM_model.t = curr_t # Assign correct time
    DNM_model = torch.load('weights/'+name+'_t='+str(round(curr_t,2)))
    optionValue[2,i] = np.transpose(DNM_model(S_plot).cpu().detach())
computing_time = (time.time() - start_time) / nSamples
print('Computing time TDGF:', computing_time)

# Plot results
Plot = cos.Plot(optionValue, labels, X_plot, name, ymax=1.0)
Plot.figures(title, SaveOutput=True)
Plot.error(times, SaveOutput=True)
Plot.error(times, SaveOutput=True, order='max')
#Plot.make_animation()

# Plot difference between option price and pay-off
diff = np.zeros((3, nSamples, n_plot))
pay_off = np.transpose(np.array(Phi(S_plot).cpu()))

for i in range(nSamples):
    for j in range(3):
        diff[j,i] = optionValue[j,i] - pay_off
        
Plot = cos.Plot(diff, labels, X_plot, name+'_diff', ymin=-0.01, ymax=0.12)
Plot.figures(title, SaveOutput=True)