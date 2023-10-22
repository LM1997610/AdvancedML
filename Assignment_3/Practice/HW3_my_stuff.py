

from utils import h36motion3d as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.loss_funcs import *
from models.sttr.sttformer import Model
from utils.data_utils import define_actions

import wandb
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import os
import time
import numpy as np
from tqdm.auto import tqdm 
from IPython.display import clear_output
import matplotlib.pyplot as plt


config_diz = {"n_epochs": 41, 
              "lr" : 1e-01, 
              "weight_decay": 1e-05,
              "milestones":list(range(3, 42, 3)), 
              "batch_size": 256, 
              "dataset":'h36m', "model": "baseline"}


class train_my_model():

    def __init__(self, config_diz, number :int):

        self.input_n = 10
        self.output_n = 25
        self.skip_rate = 1
        self.path =  './data/h3.6m/h3.6m/dataset'
        self.model_name =  'h36m'+'_3d_'+str(self.output_n)+'frames_ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_epochs = config_diz["n_epochs"]
        self.lr = config_diz["lr"]                      # learning rate
        self.weight_decay = config_diz["weight_decay"]  # weight decay (L2 penalty)
        self.milestones = config_diz["milestones"]      # the epochs after which the learning rate is adjusted by gamma

        self.batch_size = config_diz["batch_size"] 
        self.joints_to_consider = 22
        self.n_heads = 1

        self.number = number
        self.path_to_save = f'./checkpoints_v{str(number)}/'  
        self.batch_size_test = 8
        wandb.login()
      
    def make_the_model(self):

        # Load Data
        print('\nLoading Train Dataset...')
        dataset = datasets.Datasets(self.path, self.input_n, self.output_n, self.skip_rate, split=0)
        print('Loading Validation Dataset...')
        vald_dataset = datasets.Datasets(self.path, self.input_n, self.output_n, self.skip_rate, split=1)

        print('\n>>> Training dataset length: {:d}'.format(dataset.__len__()))
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)#
        print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()), "\n")
        self.vald_loader = DataLoader(vald_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        self.model = Model(num_joints=self.joints_to_consider,
                     num_frames = self.input_n, num_frames_out=self.output_n, num_heads = self.n_heads,
                     num_channels=3, kernel_size=[3,3], use_pes=True).to(self.device)
        
        # Arguments to setup the optimizer

        # lr = 1e-01                 # learning rate
        use_scheduler = True         # use MultiStepLR scheduler
        gamma = 0.1                  # gamma correction to the learning rate, after reaching the milestone epochs
        
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)

        if use_scheduler:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=gamma)


    def do_everything(self, hyper_params):

        with wandb.init(project="AML-HW3", config= hyper_params):
        
            self.make_the_model() 
            
            # launch training
            self.train()
            
            # Change the epoch according to the validation curve :
            ckpt_path = f'./checkpoints_v{self.number}/h36m_3d_25frames_ckpt_epoch_25.pt'
            self.test(ckpt_path)

        return self.model 
    
    def do_my_plot_and_save(self, my_model, train_loss, val_loss, this_epoch):

        #if not exists(path_to_save_model): makedirs(path_to_save_model)
        if not os.path.exists(self.path_to_save+ "plots_dir/"): os.makedirs(self.path_to_save + "plots_dir/")

        torch.save(my_model.state_dict(), self.path_to_save + self.model_name + "_epoch_"+str(this_epoch+1)+".pt")

        fig = plt.figure(figsize=(5, 2))
        fig.tight_layout(pad = 2)

        x_lenght = list(range(1, len(train_loss)+1))

        plt.plot(x_lenght , train_loss, 'r', label = 'Train loss')
        plt.plot(x_lenght , val_loss, 'g', label =' Val loss')

        plt.title('\n Loss History \n', fontsize=14)
        plt.xlabel('n_of_epochs \n'); plt.ylabel('loss')

        t = 1 if this_epoch < 11 else 2 if this_epoch<21 else 3
        plt.xticks(list(range(1, len(train_loss)+1, t)));
        plt.grid(linewidth=0.4); plt.legend()

        plt.savefig(self.path_to_save + "plots_dir/" +"loss_epoch_"+str(this_epoch+1)+".png", bbox_inches='tight')
        plt.show()

    def train(self, save_and_plot = True):
        
        wandb.watch(self.model, log="all", log_freq=10)
        train_loss = []
        val_loss = []
        val_loss_best = 1000

        use_scheduler = True
        clip_grad = None 
        log_step = 200

        dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                            26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                            46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                            75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

        for epoch in tqdm(range(self.n_epochs-1), position=0, leave=True):

            running_loss=0
            n=0
            self.model.train()
            example_ct = 0
            for cnt,batch in enumerate(self.data_loader):

                batch=batch.float().to(self.device)
                batch_dim=batch.shape[0]
                n+=batch_dim

                example_ct += len(batch)
                sequences_train=batch[:, 0:self.input_n, dim_used].view(-1,self.input_n,len(dim_used)//3,3).permute(0,3,1,2)
                sequences_gt=batch[:, self.input_n:self.input_n+self.output_n, dim_used].view(-1,self.output_n,len(dim_used)//3,3)

                self.optimizer.zero_grad()
                sequences_predict=self.model(sequences_train).view(-1, self.output_n, self.joints_to_consider, 3)

                loss = mpjpe_error(sequences_predict,sequences_gt)

                if cnt % log_step == 0:
                    print('[Epoch: %d, Iteration: %5d]  training loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))

                loss.backward()

                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

                self.optimizer.step()
                running_loss += loss*batch_dim

            wandb.log({"epoch": epoch, "loss": running_loss.detach().cpu()/n}, step =epoch)
            train_loss.append(running_loss.detach().cpu()/n)

            self.model.eval()
            with torch.no_grad():

                running_loss=0
                n=0

                for cnt,batch in enumerate(self.vald_loader):

                    batch=batch.float().to(self.device)
                    batch_dim=batch.shape[0]
                    n+=batch_dim

                    sequences_train=batch[:, 0:self.input_n, dim_used].view(-1,self.input_n,len(dim_used)//3,3).permute(0,3,1,2)
                    sequences_gt=batch[:, self.input_n:self.input_n+self.output_n, dim_used].view(-1,self.output_n,len(dim_used)//3,3)

                    sequences_predict=self.model(sequences_train).view(-1, self.output_n, self.joints_to_consider, 3)
                    loss=mpjpe_error(sequences_predict,sequences_gt)

                    if cnt % log_step == 0:
                            print('[Epoch: %d, Iteration: %5d]  validation loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))

                    running_loss+=loss*batch_dim

                wandb.log({"epoch": epoch, "val_loss": running_loss.detach().cpu()/n}, step =epoch)
                val_loss.append(running_loss.detach().cpu()/n)

                if running_loss/n < val_loss_best:
                    val_loss_best = running_loss/n

            if use_scheduler:
                self.scheduler.step()

            # save and plot model every 5 epochs
            # Insert your code below. Use the argument path_to_save_model to save the model to the path specified.

            if save_and_plot and epoch in [4 + 5 * i for i in range(self.n_epochs)]:

                clear_output(wait=True)
                self.do_my_plot_and_save(self.model, train_loss, val_loss, epoch )

    def test(self, ckpt_path=None):

        self.model.load_state_dict(torch.load(ckpt_path))
        print('\n ...model loaded \n')
        self.model.eval()

        accum_loss = 0
        n_batches = 0     # number of batches for all the sequences
        actions = define_actions('all')
        dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                            26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                            46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                            75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
        # joints at same loc
        joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        joint_equal = np.array([13, 19, 22, 13, 27, 30])
        index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

        totalll=0
        counter=0

        for action in actions:

            running_loss=0
            n=0
            dataset_test = datasets.Datasets(self.path, self.input_n,self.output_n,self.skip_rate, split=2,actions=[action])
            #print('>>> test action for sequences: {:d}'.format(dataset_test.__len__()))

            test_loader = DataLoader(dataset_test, batch_size=self.batch_size_test, shuffle=False, num_workers=0, pin_memory=True)

            for cnt,batch in enumerate(test_loader):
                with torch.no_grad():

                    batch=batch.to(self.device)
                    batch_dim=batch.shape[0]
                    n+=batch_dim

                    all_joints_seq=batch.clone()[:, self.input_n:self.input_n+self.output_n,:]

                    sequences_train=batch[:, 0:self.input_n, dim_used].view(-1,self.input_n,len(dim_used)//3,3).permute(0,3,1,2)
                    sequences_gt=batch[:, self.input_n:self.input_n+self.output_n, :]

                    running_time = time.time()
                    #sequences_predict = model(sequences_train)
                    sequences_predict = self.model(sequences_train).view(-1, self.output_n, self.joints_to_consider, 3)

                    totalll += time.time()-running_time
                    counter += 1

                    sequences_predict = sequences_predict.contiguous().view(-1,self.output_n,len(dim_used))

                    all_joints_seq[:,:,dim_used] = sequences_predict
                    all_joints_seq[:,:,index_to_ignore] = all_joints_seq[:,:,index_to_equal]

                    loss = mpjpe_error(all_joints_seq.view(-1,self.output_n,32,3),sequences_gt.view(-1,self.output_n,32,3))
                    running_loss += loss*batch_dim
                    accum_loss += loss*batch_dim

            print(str(action),': ', str(np.round((running_loss/n).item(),1)))
            n_batches += n

        print('\nAverage: '+str(np.round((accum_loss/n_batches).item(),1)))
        print('Prediction time: ', totalll/counter)
        wandb.log({"test_mpjpe_error" : np.round((accum_loss/n_batches).item(),1) })
        


