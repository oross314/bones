import numpy as np
import torch
import torch.nn as nn
from torch import optim
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys, os
import warnings
import copy 
from sklearn.metrics import confusion_matrix

weight_path = 'weights'
val_file = 'params.p'



class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__()
        """Initializes the network.

        Args:
            sizes: (list of integers) the size of the input data
            lins: (list of integers) the sizes of the linear layers
            activation: (torch.nn or custom activation function) the activation function to use
        Kwargs:
            convs: (list of integers) the number of convolution channels in each layer
            csizes: (list of integers) the size of the convolution kernel in each layer
            dropout: (float, default 0) the dropout probability
            resnet: (bool, default False) whether to use resnet architecture

        Returns:
            None
        """
        sizes, lins, self.activation = args 
        
        
        self.convs = kwargs.get('convs', [])
        csizes = kwargs.get('csizes', [])
        dropout = kwargs.get('dropout', 0)
        self.resnet = kwargs.get('resnet', False)
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

        ##Define convolution size to use
        if len(sizes) == 1: conv_layer = nn.Conv1d
        elif len(sizes) == 2: conv_layer = nn.Conv2d
        elif len(sizes) == 3: conv_layer = nn.Conv3d

        ## Create List of Convolution Layers
        if len(self.convs):
            self.convs = np.append([1,], self.convs)
            self.cfcs = nn.ModuleList()        
            for i in range(len(self.convs)-1):
                self.cfcs.append(conv_layer(self.convs[i], self.convs[i+1], csizes[i], padding = pads[i]).double())

        ## Create List of Linear Layers
        self.lfcs = nn.ModuleList()
        if lins:
            lins = np.append([lsize,], lins)
            for i in range(len(lins)-1):
                self.lfcs.append( nn.Linear(lins[i], lins[i+1], ))

        ## create a list of Activations
        if self.activation:
            self.acts = nn.ModuleList()
            for i in range(len(lins)-1):
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
            output = output.flatten(start_dim = 1)
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
        """Initializes the model object

        Args:
            lins: (list of integers) the sizes of the linear layers
            activation: (torch.nn or custom activation function) the activation function to use
            optim: (torch.optim) the optimizer to use
            batch_size: (int) the batch size
            lr: (float) the initial learning rate
            indata: (numpy array) the input data
            truth: (numpy array) the truth data
            cost: (torch.nn or custom loss function) the loss function to use

        kwargs:
            convs: (list of integers) the number of convolution channels in each layer
            csizes: (list of integers) the size of the convolution kernel in each layer
            lr_decay: (float, default 1) the learning rate decay
            saving: (bool, default False) whether to save the model
            run_num: (int) the run number to continue from
            new_tar: (bool, default False) whether to start saving to a new file
            save_weights: (bool, default False) whether to save the model weights to a .tar file
            lr_min: (float, default 1e-10) the minimum learning rate
            dropout: (float, default 0) the dropout probability
            resnet: (bool, default False) whether to use resnet architecture
            train_frac: (float, default 4/5) the fraction of the data to use for training
            test_set: (list of integers) the indices of the data to use for testing
            cov: (numpy array, default 1) the covariance matrix of the truth values
            max_batch: (int, default all) the maximum number of batches to use per epoch
            check: (bool, default True) whether to check for other runs with the same parameters
            threads: (int, default 1) the number of threads to use



        Returns:
            None
        """
        self.lins, self.activation, self.optim,  self.batch_size, self.lr, self.indata, self.truth, self.cost =args
        
        keys = ('convs', 'csizes','lr_decay',  'saving', 'run_num', 'new_tar', 'save_weights',
                'lr_min','dropout', 'resnet', 'train_frac', 'test_set', 'cov', 'max_batch', 'check', 'threads' )
        for kwarg in kwargs.keys():
            if kwarg not in keys:
                raise Exception(kwarg + ' is not a valid key. Valid keys: ', keys )
        torch.set_num_threads(int(kwargs.get('threads', 1)))
        self.cov = kwargs.get('cov', 1)
        self.convs = kwargs.get('convs', [])
        self.csizes = kwargs.get('csizes', [])
        self.resnet = kwargs.get('resnet', False)
        if self.csizes and self.convs :
            if len(self.csizes) != len(self.convs):
                raise Exception('Number of convolutions does not match the number of convolution sizes')
        elif self.csizes or self.convs:
            raise Exception('Provide the number of convolution channels and the size of the convolution matrix')
            
       
        self.train_frac = kwargs.get('train_frac', 4/5)
        self.test_set = kwargs.get('test_set', [])
        if len(self.test_set):
                self.test_set = (self.test_set,)
        if not self.train_frac == 4/5 and self.test_set:
            warnings.warn('Specifying a training set supercedes specifying a training fraction')
        
        
        self.data = self.data_prep()
        self.datadims = len(self.indata.shape) - 1
                
        self.dropout = kwargs.get('dropout', 0)
        self.net = Net(self.indata.shape[1:], self.lins,  self.activation, convs = self.convs, 
        csizes = self.csizes, dropout = self.dropout, resnet = self.resnet).double()
        
        self.init_lr = self.lr
        self.lr_decay = kwargs.get('lr_decay', 1)
        self.lr_min = kwargs.get('lr_min', 1e-10)
        self.optimizer = self.optim(self.net.parameters(), lr = self.lr)
    
            
        self.trainerr, self.testerr, self.err, self.epoch, self.loc, self.save_file = [], [], 0, 0, 0, None #just inits
        self.plotev = 10
        
        self.saving = kwargs.get('saving', False)
        self.save_tar = kwargs.get('save_weights', False)
        
        if self.save_tar and not os.path.isdir(weight_path):
            os.makedirs(weight_path)
        
        if self.save_tar and not self.saving:
            raise Exception('Saving must be enabled to save model weights')
        if self.save and not os.path.exists(val_file):
            pickle.dump(self.params(), open(val_file, 'wb'))
            
        self.run_num = kwargs.get('run_num', None)
        self.new_tar = kwargs.get('new_tar', False)
        if kwargs.get('check', True):
            self.check()
        if self.run_num:
            vals = pickle.load(open(val_file,'rb')).loc[self.run_num]
            self.epoch = vals['epochs']
            self.err = vals['test loss']
            self.lr = vals['learning rate']
            print('Weights from run ', str(self.run_num), ' loaded.')
        else: self.epoch = 0      

        self.batches = int(np.floor(self.data[1].shape[0]/self.batch_size))
        self.max_batch = kwargs.get('max_batch', self.batches)
        print(self.max_batch)
  
            
        
    def params(self, returns = True):
        """This function returns the parameters of the model.
        Kwargs:
            returns: (bool) returns parameters when True, displays when False

        Returns:
            None        
        """

        if not len(self.testerr):
            trainerr = None
            testerr = None
        else:
            trainerr = round(float(self.trainerr[-1]), 2)
            testerr = round(float(self.testerr[-1]), 2)
            

        
            
        df = pd.DataFrame(columns = ('conv channels', 'conv kernel sizes',  'linear layer sizes', 'activation', 'Dropout', 'ResNet',
                              'training sample size', 'optimizer',  'batch size', 'initial learning rate',   'learning rate decay',
                                     'learning rate',  'epochs',  'train loss', 'test loss', ))
        vals = ( self.convs, self.csizes,  self.lins, str(self.activation).split('.')[-1][:-2],  self.dropout,  self.resnet ,
                    self.data[3].shape[0],  str(self.optimizer).split()[0], 
                self.batch_size,  self.init_lr,  self.lr_decay, self.lr, self.epoch,  trainerr, testerr,)
        df.loc[self.loc] = vals
        if returns == True:
            return df
        else: display(df)
            
            
    def check(self):
        """This function checks if the current parameters match any previous runs. """
        
        df = self.params()
        if os.path.exists(val_file):
            vals = pickle.load(open(val_file, 'rb'))
        else:
            vals = df
            pickle.dump(vals, open(val_file, 'wb'))
        
        ## print warning and run params if current parameters match other runs
        same = np.array([])
        for i in list(vals.index):
            if vals.loc[i][:-4].equals(df.loc[0][:-4]):
                same = np.append(same, i)
                if not i==self.run_num:
                    display(pd.DataFrame(vals.loc[i]).T)
        if self.run_num and len(same):
            same = list(np.delete(same, np.where(same == self.run_num)))
        if len(same):
            warnings.warn('Run(s) '+ str(same) + ' used the same hyper parameters')
            
        # if saving to a new file, load weights. If continuing run, check that vals match
        if self.new_tar and type(self.run_num)==int:
            self.net.load_state_dict(torch.load(weight_path + '/' + str(self.run_num)+'.tar'))        
        elif not self.new_tar and self.run_num:
            if vals.loc[self.run_num][:5].equals(df.loc[self.loc][:5]):
                self.loc = self.run_num
                self.net.load_state_dict(torch.load(weight_path + '/' + str(self.loc)+'.tar'))   
            else: raise Exception('Parameters don\'t match')

                     
        if not self.loc: 
            self.loc = vals.index[-1] + 1
    
        if not self.save_tar:
            self.save_file = None
        else: 
            self.save_file = weight_path + '/' + str(self.loc)+'.tar'
            print(f'saving weights to {self.save_file}')
            
    def save(self): 

        """This function saves the model parameters."""
        

        if self.save_tar:
            torch.save(self.net.state_dict(), open(self.save_file, 'wb'))
        vals = pickle.load(open(val_file, 'rb'))
        df = self.params()
        if self.loc in vals.index.tolist():
            vals.loc[self.loc] = df.loc[self.loc]
        else:
            vals = pd.concat([vals, df])
        pickle.dump(vals, open(val_file, 'wb'))
        
  
        
                
    def run(self, **kwargs):
        """This function runs training and testing of the model.

        Kwargs:
            lr: (float) learning rate
            lr_decay: (float) learning rate decay per epoch
            lr_min: (float) minimum learning rate
            epochs: (int) number of epochs to run
            training: (bool) whether to train the model
            plot_ev: (bool) number of epochs between plots and loss reports

        Returns:
            None
        """
        if len(self.trainerr)!=len(self.testerr):
            self.trainerr = self.trainerr[:-1]
        keys = ('lr', 'lr_decay', 'lr_min', 'epochs', 'training', 'plot_ev')
        for kwarg in kwargs.keys():
            if kwarg not in keys:
                raise Exception(kwarg + ' is not a valid key. Valid keys: ', keys )
                
        train, traintruth, traincov, test, testtruth, testcov = self.data
        self.epochs = kwargs.get('epochs', 20)
        self.lr = kwargs.get('lr', self.lr)
        self.lr_decay = kwargs.get('lr_decay', self.lr_decay)
        self.lr_min = kwargs.get('lr_min', self.lr_min)
        training = kwargs.get('training', True)
        self.plotev = kwargs.get('plot_ev', self.plotev)
        e0 = self.epoch
        batches = round(len(traintruth)/self.batch_size)
        
        while self.epoch - e0 < self.epochs:
            l = 0
            shuffle = torch.randperm(len(traintruth)) #shuffle training set
            self.lr = max(self.lr * self.lr_decay, self.lr_min)  #lr decay
            for i in range(self.max_batch):
                self.optimizer.param_groups[0]['lr'] = self.lr
                where = shuffle[i * self.batch_size:(i + 1) * self.batch_size] #take batch of training set
                self.output = self.net(train[where])
                self.truetrain = traintruth[where]
                if type(self.cov)== int:
                    loss = torch.mean(self.cost(self.output.squeeze().squeeze(), torch.tensor(traintruth[where])))
                else:
                    loss = torch.mean(self.cost(self.output.squeeze().squeeze(), torch.tensor(traintruth[where]), traincov))
                l+= loss
                if training: #turn off training to run without optimization
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph = True)
                    self.optimizer.step()
                         
            self.trainerr.append((l/batches).detach().numpy())
            self.testout = self.net(test)
            if type(self.cov) == int:
                self.err = self.cost(torch.squeeze(self.testout), torch.tensor(testtruth)).mean().detach().numpy()
            else:
                self.err = self.cost(torch.squeeze(self.testout), torch.tensor(testtruth), testcov).mean().detach().numpy()
            self.testerr.append(self.err)

            if self.plotev:
                if not (self.epoch-e0) % self.plotev:
                    self.plot()
                    print(f"Epoch number {self.epoch}:")
                    print(f" Test Loss: {np.round(self.testerr[-1], 3)}; Train Loss {np.round(self.trainerr[-1], 3)}")

                    if not training:
                        return []

            t = 10
            if not self.epoch%t or self.trainerr[-1]!=self.trainerr[-1]:   
                self.checkpoint(t)
                
            if self.saving:
                self.save()
            self.epoch += 1
    
    def checkpoint(self, t = 10): 
        """ This function checks if the training is going well.
         If the current train loss is nan, inf, or >5x the loss at the previous checkpoint, 
         it reverts to the last checkpoint.

         Kwargs:
            t: (int) the number of epochs between checkpoints

        """

        if len(self.trainerr) < t +1:
            return 0
        if not self.epoch>t:
            #create a checkpoint
            self.oldmodel = copy.deepcopy(self.net.state_dict())#create a checkpoint to return to if training goes off the rails
            self.oldlr = 1*self.lr
            self.countcheck = 0
        elif (self.trainerr[-1]/self.trainerr[-1 - t]>5) or (self.trainerr[-1]!=self.trainerr[-1]):
            #return to checkpoint if necessary
            print('Reverting to checkpoint')
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
            



            
    def plot(self):
        """ This function plots the results of the training and testing. """
        testtruth = self.data[5]
        output = self.testout
        if len(self.testerr) > 1:
            fig, ax = plt.subplots(1,3, figsize = (15,4))
            xs = np.arange(len(self.trainerr))
            ax[2].plot(xs, self.trainerr, label = 'Train loss')
            ax[2].plot(xs, self.testerr, label = 'Test loss')
            ax[2].set_xlabel('Epoch')
            ax[2].set_ylabel('Loss')
            ax[2].legend()
            ax[2].set_yscale('log')
        else: fig, ax = plt.subplots(1, 2, figsize = (15, 4))
            
        x, y = self.data[4], self.testout.flatten(start_dim = 1).detach().numpy()
        yt, xt = self.net(self.data[0]).flatten(start_dim = 1).detach().numpy(), self.data[1]
        
         
        ax[1].set_xlabel('test truth')
        ax[1].set_ylabel('test predication')
        ax[0].set_xlabel('train truth')
        ax[0].set_ylabel('train predication')
        
        if str(self.cost) == 'CrossEntropyLoss()':
            classes = np.unique(self.truth)
            num_classes = len(classes)
            cm = []
            
            cm.append(confusion_matrix(xt, classes[np.argmax(yt, axis = 1)]))
            cm.append(confusion_matrix(x, classes[np.argmax(y, axis = 1)]))
            for i in (0, 1):
                ax[i].set_xticks(np.arange(num_classes))
                ax[i].set_yticks(np.arange(num_classes))
                ax[i].set_xticklabels(classes)
                ax[i].set_yticklabels(classes)
                cbar = ax[i].figure.colorbar(ax[i].imshow(cm[i], cmap='Blues'), ax=ax[i])

                # Display values inside the plot
                thresh = cm[i].max() / 2
                for k in range(num_classes):
                    for j in range(num_classes):
                        ax[i].text(j, k, cm[i][k, j], ha='center', va='center', color='white' if cm[i][k, j] > thresh else 'black')
        else:
            ax[0].hist2d(xt, np.squeeze(yt), bins = 30)
            ax[1].hist2d(x, np.squeeze(y), bins = 30)
            
        fig.tight_layout()
        plt.show()
    
    def data_prep(self):
        """ This function prepares the data for training and testing. """
        data = torch.tensor(self.indata).double()
        
        if len(data.shape) == 1:
            data = torch.unsqueeze(data, 0)
        if len(data.shape) >= 2:
            data = torch.unsqueeze(data, 1)

        if self.test_set:
            
            test, testtruth = data[self.test_set], self.truth[self.test_set]
            trainwhere = [k for k in range(data.shape[0]) if not np.any(k == self.test_set)]
            train, traintruth = data[trainwhere], self.truth[trainwhere]
        else:
            split = int(self.train_frac * self.truth.shape[0])
            train, traintruth = data[:split], self.truth[:split]
            test, testtruth = data[split:], self.truth[split:]
            
        if type(self.cov)!=int:
            cov = torch.tensor(self.cov).double()
            traincov = cov[:split]
            testcov = cov[split:]
        else:
            traincov, testcov = 1, 1
        return train, traintruth, traincov, test, testtruth, testcov