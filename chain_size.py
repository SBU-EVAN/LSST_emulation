#!/usr/bin/env python
# coding: utf-8

# ## Chain size
# The goal of this notebook is to examine how many samples are needed for the normalizing flow result to converge and to get a good estimate of the evidence. 
# 
# There is a chain (godzilla.h5) with 120 walkers and 200000 samples per walker for a total of 24000000 samples. The analysis will be done by decreasing the thinning parameter of the chain and repeating the calculations.

# In[1]:


import numpy as np

import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from getdist import plots, MCSamples
import getdist
from multiprocessing import Pool
from getdist import plots, MCSamples, WeightedSamples

import sys
import time
from cocoa_emu import *
from cocoa_emu.emulator import NNEmulator, GPEmulator
from cocoa_emu.data_model import LSST_3x2

import emcee
import time

# Now normalizing flow
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tf.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import Callback

from numpy import linalg
import scipy

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


### open the long chain
filename = "/home/grads/ownCloud/StatisticalProject/chains/godzilla.h5"
reader = emcee.backends.HDFBackend(filename, read_only=True)

#all samples
samples = reader.get_chain(flat=True)
#print(len(samples))
#remove burn in and thin
samples_thin = reader.get_chain(flat=True, thin=1000, discard=5000)


# In[3]:


### open the planck 3x2 chain in getdist
path = '/home/grads/ownCloud/StatisticalProject/chains/plikHM_TTTEEE/base_plikHM_TTTEEE'
planck = getdist.mcsamples.loadMCSamples(file_root=path, no_cache=True)


# In[4]:


### sanity check 
samples_names = ['logA', 'ns', 'H0', 'omegabh2', 'omegach2']
planck_names = planck.getParamNames().getRunningNames()
#print(samples_names)
#print(planck_names)
# get parameters only in both chains
common_params = [param for param in planck_names if param in samples_names]
#print(common_params)
idx_1 = [samples_names.index(param) for param in common_params]
idx_2 = [planck_names.index(param) for param in common_params]
#print(idx_1)
#print(idx_2)
# get the data for common params
common_samples = samples_thin[...,idx_1]
common_planck  = planck.samples[...,idx_2]
#print(common_samples)
#print(common_planck)
samples_chain_common = MCSamples(samples=common_samples,names=common_params,labels=common_params,label='LSST')
planck_chain_common = MCSamples(samples=common_planck,names=common_params,labels=common_params,label='Planck')
g = plots.get_subplot_plotter()
g.triangle_plot([samples_chain_common,planck_chain_common], filled=True, params=common_params)
#g = plots.get_subplot_plotter()
#g.triangle_plot([planck_chain_common], filled=True, params=common_params)


# In[55]:


# create an array of chain lengths:
samples = reader.get_chain(flat=True, thin=1, discard=5000)
#print(len(samples))
#chain_lengths = np.arange(100,10000,10000)
chain_lengths = [100,500,1000,5000,10000,20000,40000,60000,80000,100000,150000,200000,250000]#,500000,1000000,5000000,10000000,20000000]
#chain_lengths = np.concatenate((chain_lengths,np.array([len(samples)])))
#print(len(chain_lengths))


# In[56]:


# These are the functions needed for the flow
class Callback(tfk.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self._loss = []
        self._epoch = []
        self.n_epochs = self.params['epochs']
        print('[                    ] Training... ',end="")
        
    def on_epoch_begin(self, epoch, logs=None):
        progress = int(epoch/self.n_epochs*20)
        ret = '\r['
        for i in range(progress):
            ret += '#'
        for i in range(20-progress):
            ret += ' '
        print(ret+'] Training... (epoch {}/{})'.format(epoch,self.n_epochs),end="")

    def on_epoch_end(self, epoch, logs=None):
        self._loss.append(logs['loss'])
        self._epoch.append(epoch)

    def on_train_end(self, logs=None):
        print('\r'+'[####################] Completed!                                    ')
        fig,ax1 = plt.subplots(1,1)
        
        ax1.set_title('loss vs. epoch')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.plot(self._epoch,self._loss)
        
class No_Plot_Callback(tfk.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.n_epochs = self.params['epochs']
        print('[                    ] Training... ',end="")
        
    def on_epoch_begin(self, epoch, logs=None):
        progress = int(epoch/self.n_epochs*20)
        ret = '\r['
        for i in range(progress):
            ret += '#'
        for i in range(20-progress):
            ret += ' '
        print(ret+'] Training... (epoch {}/{})'.format(epoch,self.n_epochs),end="")

    def on_train_end(self, logs=None):
        print('\r'+'[####################] Completed!                                  ')

def pregauss(chain,data):
    covmat = chain.cov().astype(np.float32)
    mean = chain.getMeans().astype(np.float32)
    
    # bijector time!
    # TriL means the cov matrix is lower triangular. Inverse is easy to compute that way
    # the cholesky factorization takes a positive definite hermitian matrix M (like the covmat) to LL^T with L lower triangluar
    gauss_approx = tfd.MultivariateNormalTriL(loc=mean,scale_tril=tf.linalg.cholesky(covmat))
    bijector = gauss_approx.bijector
    
    new_data = bijector.inverse(data.astype(np.float32))

    # now map the data
    return new_data,bijector

def train(base,data,bijectors,batch_size,n_epochs,feedback=True):
    #covmat = data.cov().astype(np.float32)
    #mean = data.getMeans().astype(np.float32)

    val_split = 0.1
    # stack data
    _data = []
    dim = 0
    for key in data.getParamNames().list():
        nsamples=len(data[key])
        _data.append(data[key])
        dim += 1

    #print(_data)
    xdata = np.stack(_data, axis=-1)
    #print(xdata)

    x_data,bij = pregauss(data,xdata)

    #create data set with weights.
    weights = data.weights.astype(np.float32)
    
    ## NN setup
    target_distribution = tfd.TransformedDistribution(
        distribution=base,
        bijector=tfb.Chain(bijectors)) 

    # Construct model.
    x_ = tfk.Input(shape=(dim,), dtype=tf.float32)
    log_prob_ = target_distribution.log_prob(x_)
    model = tfk.Model(x_, log_prob_)

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),
                  loss=lambda _, log_prob: -log_prob) 
    if(feedback):
        print('---   Model compiled   ---')
        print(" - N samples = {}".format(nsamples))
        if weights.all()==weights[0]:
            print(" - Uniform weights = {}".format(weights[0]))
        else:
            print(" - Non-uniform weights")
        print(" - Pre-Gaussian Map = True\n")
        print(" - Validation split = {}".format(val_split))
        print(' - Number MAFS = {} '.format(int(len(bijectors)/2)))
        print(' - Trainable parameters = {} \n'.format(model.count_params()))

    # now perform the fit
    if(feedback):
        model.fit(x=x_data,
                  y=np.zeros((nsamples, dim),dtype=np.float32),
                  batch_size=batch_size,
                  epochs=n_epochs,
                  steps_per_epoch=int(nsamples/batch_size*0.8),  # Usually `n // batch_size`.
                  validation_split=val_split,
                  shuffle=True,
                  verbose=False,
                  callbacks=[Callback(),tfk.callbacks.ReduceLROnPlateau()]) #, ydata
    if(not feedback):
        model.fit(x=x_data,
                  y=np.zeros((nsamples, dim),dtype=np.float32),
                  batch_size=batch_size,
                  epochs=n_epochs,
                  steps_per_epoch=int(nsamples/batch_size*0.8),  # Usually `n // batch_size`.
                  validation_split=val_split,
                  shuffle=True,
                  verbose=False,
                  callbacks=[No_Plot_Callback(),tfk.callbacks.ReduceLROnPlateau()]) #, ydata
        
    return(target_distribution,bij)

def setup(n_maf,n_params,permute,feedback=True):
    # Set up bijector MADE
    hidden_units=[n_params*2]*2
    if(feedback):
        print('---   MADE Info   ---')
        print(' - Hidden_units = {}'.format(hidden_units))
        print(' - Activation = {}\n'.format(tf.math.asinh))
    bijectors=[]
    if(permute==True):
        _permutations = [np.random.permutation(n_params) for _ in range(n_maf)]
    else:
        _permutations=False
    
    for i in range(n_maf):
        # the permutation part comes from the code M. Raveri wrote,
        if _permutations:
            #print(_permutations[i])
            bijectors.append(tfb.Permute(_permutations[i].astype(np.int32)))
        # rest by myself
        bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=tfb.AutoregressiveNetwork(params=2, event_shape=(n_params,), hidden_units=hidden_units, activation=tf.math.asinh, kernel_initializer='glorot_uniform')))
        
    return bijectors

def diff_boost(n_boost,chains):
    ### In this notebook, the first sample is variable length and the second is planck
    #   We want to use only the number of samples that are in the variable length
    #   So the code to comput the difference is slightly changed from the final implementation
    
    # get data
    chain1 = chains[0].samples
    chain2 = chains[1].samples
    w_chain1 = chains[0].weights
    w_chain2 = chains[1].weights
    ll_chain1 = chains[0].loglikes
    ll_chain2 = chains[1].loglikes
    
    samples_names = chains[0].getParamNames().getRunningNames()
    planck_names = chains[1].getParamNames().getRunningNames()
    common_params = [param for param in planck_names if param in samples_names]
    
    idx1 = [samples_names.index(param) for param in common_params]
    idx2 = [planck_names.index(param) for param in common_params]
    assert len(idx1)==len(idx2)
    # get the data for common params
    common_samples = samples[...,idx1]
    common_planck  = planck.samples[...,idx2]
    
    # ensure first chain is longer than the second.
    # Need to keep track if I flipped the data so I get the signs right (although in principle it doesn't matter, Its better for everyones results to look the same even if they import chains in different orders)
    flip=False
    if( len(chain1) < len(chain2) ):
        #chain1,chain2 = chain2,chain1
        #w_chain1,w_chain2 = w_chain2,w_chain1
        #ll_chain1,ll_chain2 = ll_chain2,ll_chain1
        #idx1,idx2 = idx2,idx1
        flip=True
    
    N1 = len(chain1)
    N2 = len(chain2)
    #print('N1 = {}'.format(N1))
    #print('N2 = {}'.format(N2))
    
    # set up parameter diff arrays
    diff = np.zeros((N1*n_boost,len(idx1)),dtype=np.float32)
    weights = np.zeros(N1*n_boost,dtype=np.float32)
    loglikes = np.zeros(N1*n_boost,dtype=np.float32)
        
    for i in range(n_boost):
        # find the range of indices to use for chain 2
        lower = int((i/n_boost)*N1)
        upper = lower+N1

        # compute stuff
        if flip==True:
            #diff[i*N1:(i+1)*N1] = -chain1[:N1,idx1] + np.take(chain2[:,idx2], range(lower,upper), axis=0, mode='wrap')
            diff[i*N1:(i+1)*N1] = chain1[:N1,idx1] - chain2[:N1,idx2]
            weights[i*N1:(i+1)*N1] = w_chain1 * w_chain2[:N1]
        else:
            diff[i*N1:(i+1)*N1] = chain1[:N1,idx1] - np.take(chain2[:,idx2], range(lower,upper), axis=0, mode='wrap')
            weights[i*N1:(i+1)*N1] = w_chain1*np.take(w_chain2, range(lower,upper), mode='wrap')
        #weights[i*N1:(i+1)*N1] = w_chain1*np.take(w_chain2, range(lower,upper), mode='wrap')
        #if(ll_chain1 is not None and ll_chain2 is not None):
        #    loglikes[i*N1:(i+1)*N1] = ll_chain1+np.take(ll_chain2, range(lower,upper), mode='wrap')
    
    min_weight_ratio = min(chains[0].min_weight_ratio,
                               chains[1].min_weight_ratio)

    diff_samples = WeightedSamples(ignore_rows=0,
                                   samples=diff,
                                   weights=weights, loglikes=loglikes,
                                   name_tag=' ', label=' ',
                                   min_weight_ratio=min_weight_ratio)

    return diff_samples
    
def significance(trained_dist,bijector,nparams,alpha=0.32):
    # The alpha is used for beta function for the confidence. Raveri et. al. defaults to 0.32
    prob = trained_dist.prob(bijector.inverse(np.zeros(nparams,dtype=np.float32)))
    n_points = 10000
    n_pass = 0

    _s = trained_dist.sample(n_points)
    _v = trained_dist.prob(_s)
    for val in _v:
        if val>prob:
            n_pass+=1
    # use clopper-pearson to find confidence level
    low = scipy.stats.beta.ppf(alpha/2,float(n_pass),float(n_points-n_pass+1))
    high = scipy.stats.beta.ppf(1-alpha/2,float(n_pass+1),float(n_points-n_pass))

    # compute sigma based on gaussian
    n_sigma = np.sqrt(2)*scipy.special.erfinv(n_pass/n_points)
    sigma_high = np.sqrt(2)*scipy.special.erfinv(high)
    sigma_low = np.sqrt(2)*scipy.special.erfinv(low)
    
    return n_sigma,sigma_high,sigma_low


# In[83]:


# Now we want to do the NF for each chain length.

n_runs = 25 # the NF will be done n_runs times for each chain length. The uncertainty is computed from the variance of the results

N = np.zeros(len(chain_lengths))
n_sigma = np.zeros((len(chain_lengths),2))
for i in range(len(chain_lengths)):
    run = 0
    sigmas = np.zeros(n_runs+1)
    l = chain_lengths[i]
    sigmas[0]=l
    while run < n_runs:
        print('Chain length = {}\nrun {}/{}...'.format(l,run+1,n_runs))
        #idxs = np.random.choice(len(samples),size=l)
        #print(idxs)
        s = samples[:l,...]
        #s = samples[idxs,...]
        #print(s)
        
        #s = reader.get_chain(flat=True, thin=10000, discard=5000)
        #print(len(s))
        samples_chain = MCSamples(samples=s[...,:5],names=samples_names,labels=samples_names)
        diff = [diff_boost(1,[samples_chain,planck])]
        param_diff_chain = getdist.MCSamples(names=common_params,labels=common_params)
        param_diff_chain.chains = diff
        param_diff_chain.makeSingle()

        n_params = len(param_diff_chain.getParamNames().list())
        dist = tfd.MultivariateNormalDiag(
        loc=np.zeros(n_params,dtype=np.float32), 
        scale_diag=np.ones(n_params,dtype=np.float32))

        bijectors = setup(2*n_params,n_params,True,feedback=False)
        
        batch=100
        if l<1000:
            batch=10

        trained_dist,bijector = train(dist,param_diff_chain,bijectors=bijectors,batch_size=int(l/batch),n_epochs=100,feedback=False)
        sigma,high,low = significance(trained_dist,bijector,n_params)
        sigmas[run+1] = sigma
        #print(sigma)
        
        run+=1
        
    #N[i] = l
    #n_sigma[i,0] = np.mean(sigmas)
    #n_sigma[i,1] = np.std(sigmas)
    np.savetxt('sigmavN'+str(l)+'.txt',sigmas)
    
#print(n_sigma)
#print(N)


#param_diff_chain = getdist.MCSamples(names=common_params,labels=common_params)
#param_diff_chain.chains = diff
#param_diff_chain.makeSingle()
#g = plots.get_subplot_plotter()
#g.triangle_plot(param_diff_chain, params=common_params, filled=True)


# In[ ]:


#sample = MCSamples(samples=_samples.numpy(), names=common_params,label='learned dist')
#param_diff_chain = MCSamples(samples=param_diff,names=common_params,label='true dist')
#g = getdist.plots.get_subplot_plotter()
#g.settings.num_plot_contours = 2
#g.triangle_plot([param_diff_chain,sample],
#                params=common_params,
#                filled=False)


# In[ ]:


#nsigma_data = []
#for i in range(len(N)):
#    nsigma_data.append([N[i],n_sigma[i,0],n_sigma[i,1]])
#print(nsigma_data)
#np.savetxt('n_sigma_v_N_xl.txt',nsigma_data)


# In[ ]:


#data = np.array(nsigma_data)
#plt.errorbar(data[...,0],data[...,1],yerr=data[...,2],lw=0,elinewidth=2,marker='o')
#
#ax = plt.gca()
#ax.set_ylim([0, 2])


# In[ ]:


#

