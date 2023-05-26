import numpy as np
import torch
import torch.nn as nn
from torch import optim
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys, os
import warnings
import math
import copy 



class Net(nn.Module):
    def __init__(self, *args):
        super(Net, self).__init__()
        sizes, self.convs, lins, csizes,  dropout, self.resnet, self.activation = args  
        
        if self.resnet:
            pads = ((np.array(csizes)-1)/2).astype(int)
            lsize = self.convs[-1] * sizes[0]*sizes[1]
        elif self.convs:
            pads = np.zeros(len(csizes)).astype(int)
            lsize = self.convs[-1]
            for i in range(len(sizes)):
                lsize *= (sizes[i] - sum(csizes) + len(csizes))
        else: 
            lsize = 1
            for i in range(len(sizes)):
                lsize *= sizes[i]
        
        ## Create List of Convolution Layers
        if len(self.convs):
            self.convs = np.append([1,], self.convs)
            self.cfcs = nn.ModuleList()        
            for i in range(len(self.convs)-1):
                self.cfcs.append(nn.Conv2d(self.convs[i], self.convs[i+1], csizes[i], padding = pads[i]).double())

        ## Create List of Linear Layers
        self.lfcs = nn.ModuleList()
        if lins:
            lins = np.append([lsize,], lins)
            for i in range(len(lins)-1):
                self.lfcs.append( nn.Linear(lins[i], lins[i+1], ))

        ## create a list of Activations
        if self.activation:
            self.acts = nn.ModuleList()
            for i in range(len(lins)):
                self.acts.append(self.activation())
        
        if dropout:
            self.drops = nn.ModuleList()
            for i in range(len(lins)):
                self.drops.append(nn.Dropout(dropout))
        else: self.drops = []
        

    def forward(self, output):   
        if len(self.convs):
            repnum = self.convs[0]
            output = output/torch.mean(output)
            for i in range(len(self.cfcs)):
                if self.resnet:
                    old  = output.repeat(1, repnum, 1, 1) 
                    new = self.cfcs[i](output)
                    output = new/torch.mean(new) + old
                    repnum = int((self.convs[(i+2)%len(self.convs)])/self.convs[i+1])
                else:
                    output = self.cfcs[i](output)
                     
            #Create Embedding  
            output = output.view(output.size()[0], -1)
            output = torch.unsqueeze(output, 1)
                             
        #Run linear layers and activations
        for i in range(len(self.lfcs)):
            output = self.lfcs[i](output)
            if self.activation:
                output = self.acts[i](output)
            if self.drops:
                output = self.drops[i](output)

        return output

    
class Model:
    def __init__(self, *args, **kwargs):
        self.lins, self.activation, optimizer,  self.batch_size, self.lr, self.indata, self.truth, self.cost =args
        
        keys = ('convs', 'csizes','lr_decay', 'max_epochs', 'saving', 'run_num', 'new_tar',
                'lr_min','dropout',  'plot_ev', 'print_ev', 'resnet', 'train_fac', 'cov')
        for kwarg in kwargs.keys():
            if kwarg not in keys:
                raise Exception(kwarg + ' is not a valid key. Valid keys: ', keys )
        
        self.cov = kwargs.get('cov', 1)
        self.convs = kwargs.get('convs', [])
        self.csizes = kwargs.get('csizes', [])
        self.resnet = kwargs.get('resnet', False)
        if self.csizes and self.convs :
            if len(self.csizes) != len(self.convs):
                raise Exception('Number of convolutions does not match the number of convolution sizes')
        elif self.csizes or self.convs:
            raise Exception('Provide the number of convolution channels and the size of the convolution matrix')

        
        self.plotev = kwargs.get('plot_ev', 10)
        self.printev = kwargs.get('print_ev', 10)
        self.train_fac = kwargs.get('train_fac', 4/5)
        
        
        self.data = self.data_prep()
        self.datadims = len(self.indata.shape) - 1
        if self.datadims != 2 and self.convs:
            raise Exception('convolutions are only supported for 2d data')

                
        self.dropout = kwargs.get('dropout', 0)
        self.net = Net(self.indata.shape[1:], self.convs, self.lins,  self.csizes, self.dropout, 
                             self.resnet, self.activation).double()
        
        self.lr_decay = kwargs.get('lr_decay', 1)
        self.lr_min = kwargs.get('lr_min', 1e-10)
        self.optimizer = optimizer(self.net.parameters(), lr = self.lr)
        self.max_epochs = kwargs.get('max_epochs', 20)
            
        self.trainerr, self.testerr, self.err, self.epoch, self.loc, self.save_file = [], [], 0, 0, 0, None #just inits
        
        self.saving = kwargs.get('saving', False)
        
        self.new_tar = kwargs.get('new_tar', False)
        self.run_num = kwargs.get('run_num', None)
        self.check()

        
        if self.run_num:
            vals = pickle.load(open('values.p','rb')).loc[self.run_num]
            self.epoch = vals['epochs']
            self.err = vals['err']
            self.lr = vals['learning rate']
            print('Weights from run ', str(self.run_num), ' loaded.')
        else: self.epoch = 0        
            
        
    def params(self, re = True):
            
        if len(self.testerr)>1:
            derr = round(self.testerr[-2] - self.testerr[-1], 3)
        else:
            derr = 0
        
            
        df = pd.DataFrame(columns = ('convolution layer sizes', 'convolution matrix sizes',  
                                      'ResNet','linear layer sizes','activation', 'training sample size', 'learning rate',  
                             'Dropout',  'batch size',   
                            'epochs',  'derr', 'err', ))
        vals = ( self.convs, self.csizes,  self.resnet , self.lins, str(self.activation).split('.')[-1][:-2],
                int(self.train_fac*self.indata.shape[0]), self.lr, self.dropout,  self.batch_size,  
                    self.epoch + len(self.testerr),  derr, round(float(self.err),2),)
        df.loc[self.loc] = vals
        if re == True:
            return df
        else: display(df)
            
            
    def check(self):
        
        df = self.params()
        if os.path.exists('values.p'):
            vals = pickle.load(open('values.p', 'rb'))
        else:
            vals = df
            pickle.dump(vals, open('values.p', 'wb'))
        
        ## print warning and run params if current parameters match other runs
        same = np.array([])
        for i in list(vals.index):
            if vals.loc[i][:5].equals(df.loc[0][:5]):
                same = np.append(same, i)
                if not i==self.run_num:
                    display(pd.DataFrame(vals.loc[i]).T)
        if self.run_num and len(same):
            same = list(np.delete(same, np.where(same == self.run_num)))
        if len(same):
            warnings.warn('Run(s) '+ str(same) + ' used the same hyper parameters')
            
        # if saving to a new file, load weights. If continuing run, check that vals match
        if self.new_tar and type(self.run_num)==int:
            self.net.load_state_dict(torch.load('tars/' + str(self.run_num)+'.tar'))        
        elif not self.new_tar and self.run_num:
            if vals.loc[self.run_num][:5].equals(df.loc[self.loc][:5]):
                self.loc = self.run_num
                self.net.load_state_dict(torch.load('tars/' + str(self.loc)+'.tar'))   
            else: raise Exception('Parameters don\'t match')

                     
        if not self.loc: 
            self.loc = vals.index[-1] + 1
    
        if not self.saving:
            self.save_file = None
            print('Not saving')
        else: 
            self.save_file = 'tars/' + str(self.loc)+'.tar'
            
    def save(self): 
        torch.save(self.net.state_dict(), self.save_file)
        vals = pickle.load(open('values.p', 'rb'))
        df = self.params()
        if self.loc in vals.index.tolist():
            vals.loc[self.loc] = df.loc[self.loc]
        else:
            vals = vals.append( df)
        pickle.dump(vals, open('values.p', 'wb'))
        
  
        
                
    def run(self, **kwargs):
        if self.max_epochs ==self.epoch:
            print('maximum epoch reached')
        if len(self.trainerr)!=len(self.testerr):
            self.trainerr = self.trainerr[:-1]
        keys = ('lr', 'lr_decay', 'max_epochs', 'saving', 'batch_size', 'lr_min', 'training',)
        for kwarg in kwargs.keys():
            if kwarg not in keys:
                raise Exception(kwarg + ' is not a valid key. Valid keys: ', keys )
                
        train, traintruth, traincov, test, testtruth, testcov = self.data
        self.max_epochs = kwargs.get('max_epochs', self.max_epochs)
        self.lr = kwargs.get('lr', self.lr)
        self.saving = kwargs.get('saving', self.saving)
        self.lr_decay = kwargs.get('lr_decay', self.lr_decay)
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        self.lr_min = kwargs.get('lr_min', self.lr_min)
        training = kwargs.get('training', True)
        self.printev = kwargs.get('print_ev', self.printev)
        self.plotev = kwargs.get('plot_ev', self.plotev)
        e0 = self.epoch
        
        while self.epoch + e0 < self.max_epochs:
            shuffle = torch.randperm(len(traintruth)) #shuffle training set
            self.lr = max(self.lr * self.lr_decay, self.lr_min)  #lr decay
            for i in range(round(len(traintruth)/self.batch_size)):
                self.optimizer.param_groups[0]['lr'] = self.lr
                where = shuffle[i * self.batch_size:(i + 1) * self.batch_size] #take batch of training set
                self.output = self.net(train[where])
                self.truetrain = traintruth[where]
                if type(self.cov)== int:
                    loss = torch.mean(self.cost(self.output.squeeze().squeeze(), torch.tensor(traintruth[where]).double()))
                else:
                    loss = torch.mean(self.cost(self.output.squeeze().squeeze(), torch.tensor(traintruth[where]).double(), traincov))
                    
                if training: #turn off training to run without optimization
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph = True)
                    self.optimizer.step()
                         
            self.trainerr.append(loss.detach().numpy())
            self.testout = self.net(test)
            if type(self.cov) == int:
                self.err = self.cost(torch.squeeze(self.testout), torch.tensor(testtruth)
                                         ).mean().detach().numpy()
            else:
                self.err = self.cost(torch.squeeze(self.testout), torch.tensor(testtruth), testcov
                                         ).mean().detach().numpy()
            self.testerr.append(self.err)
            if not (self.epoch-e0) % self.plotev:
                self.plot()
                if not training:
                    return []

            if not (self.epoch - e0) % self.printev:
                print("Epoch number {}\n Current loss {}\n".format(self.epoch, self.err))
                
            t = 10
            if not self.epoch%t or self.trainerr[-1]!=self.trainerr[-1]:   
                self.checkpoint(t)
                
            if self.saving:
                self.save()
            self.epoch += 1
    
    def checkpoint(self, t = 10): 
        if not self.epoch>t:
            #create a checkpoint
            self.oldmodel = copy.deepcopy(self.net.state_dict())#create a checkpoint to return to if training goes off the rails
            self.oldlr = 1*self.lr
            self.countcheck = 0
        elif (self.trainerr[-1]/self.trainerr[-1 - t]>5) or (self.trainerr[-1]!=self.trainerr[-1]):
            #return to checkpoint if necessary
            print('rewind', float(self.trainerr[-1]), float(self.trainerr[-1 - t]), self.lr)
            self.lr = 1* self.oldlr #revert lr   
            self.net.load_state_dict(self.oldmodel) #revert weights
            #remove record of bad epochs
            self.testerr = self.testerr[:-t]  
            self.trainerr = self.trainerr[:-t]
            self.epoch -= t
            self.optimizer = self.optim(self.net.parameters(), lr = self.lr) #delete adam's memory of the oopsie
            self.countcheck+=1
        else: 
            #create a checkpoint
            self.oldmodel = copy.deepcopy(self.net.state_dict())#create a checkpoint to return to if training goes off the rails
            self.oldlr = 1*self.lr
            self.countcheck = 0 
            



            
    def plot(self, med = True, log = True):
        ###make log an input
        testtruth = self.data[5]
        output = self.testout
        if len(self.testerr) > 1:
            fig, ax = plt.subplots(1,3, figsize = (15,4))
            xs = np.arange(len(self.trainerr))
            ax[2].plot(xs, self.trainerr, label = 'train error')
            ax[2].plot(xs, self.testerr, label = 'test error')
            ax[2].set_xlabel('epoch')
            ax[2].set_ylabel('mean abs err')
            ax[2].legend()
            ax[2].set_yscale('log')
        else: fig, ax = plt.subplots(1, 2, figsize = (15, 4))
            
        x, y = self.data[4], np.squeeze(np.squeeze(self.testout.detach().numpy()))
        
        if not log:
            bins = 30
        else:
            bins = np.logspace(np.log10(min(x)), np.log10(max(x)), 30)
            ax[0].loglog()
            ax[1].loglog()
        ax[0].hist2d(x, y, bins = bins)

        ax[0].set_xlabel('test truth')
        ax[0].set_ylabel('test predication') 

        yt, xt = torch.squeeze(torch.squeeze(self.output)).detach().numpy(), self.truetrain
        ax[1].hist2d(xt, yt, bins = bins)
        ax[1].set_xlabel('train truth')
        ax[1].set_ylabel('train predication') 
        plt.show()
    
    def data_prep(self):
        data = torch.tensor(self.indata).double()
        
        if len(data.shape) == 1:
            data = torch.unsqueeze(data, 0)
        if len(data.shape) >= 2:
            data = torch.unsqueeze(data, 1)

        split = int(self.train_fac * self.truth.shape[0])
        train, traintruth = data[:split], self.truth[:split]
        test, testtruth = data[split:], self.truth[split:]
        if type(self.cov)!=int:
            cov = torch.tensor(self.cov).double()
            traincov = cov[:split]
            testcov = cov[split:]
        else:
            traincov, testcov = 1, 1
        return train, traintruth, traincov, test, testtruth, testcov

            
                            