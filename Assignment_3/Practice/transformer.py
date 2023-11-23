import torch
import torch.nn as nn
#import torch.nn.functional as F
#from torch.nn.modules import linear
from torch.autograd import Variable
from utils.loss_funcs import mpjpe_error
import numpy as np 
import math

import matplotlib.pyplot as plt
from IPython.display import clear_output
from os import makedirs
from os.path import exists

from copy import deepcopy


class Attention(nn.Module):
    # Scaled Dot-Product Attention

    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, key, value, mask=None):

        # add here the code regarding the argument of the softmax function as defined above

        scaling_factor = np.sqrt(query.size(-1))
        attn = torch.matmul(query, key.transpose(-2,-1)) / scaling_factor

        if mask is not None:

            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(nn.functional.softmax(attn, dim=-1))

        # computed attn, calculate the final output of the attention layer

        output = torch.matmul(attn, value)

        return output, attn
    


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, d_model, dropout=0.1):

        # Take in model size and number of heads.

        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        #  We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.query_ff = nn.Linear(d_model, d_model)
        self.key_ff = nn.Linear(d_model, d_model)
        self.value_ff = nn.Linear(d_model, d_model)
        self.attn_ff = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.attention = Attention(attn_dropout=dropout)

    def forward(self, query, key, value, mask=None, return_attention=False):

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k.
        # The query is given as example, you should do the same for key and value
        query = self.query_ff(query).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Add your code below

        key = self.key_ff(key).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_ff(value).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        # Add your code below

        x, self.attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)

        if return_attention:
            return self.attn_ff(x), self.attn

        return self.attn_ff(x)
    

class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):

        """Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers """

        super().__init__()

        # Attention layer
        self.self_attn = MultiHeadAttention(num_heads, input_dim)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim))

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        # Self_attention part (use self.norm1)
        # Add your code below

        attn_output = self.self_attn(x, x, x)
        x = x + self.dropout(self.norm1(attn_output))

        # MLP part (use self.norm2)
        # Add your code below

        linear_output = self.linear_net(x)
        x = x + self.dropout(self.norm2(linear_output))

        return x
    
class DecoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):

        """ Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers """

        super().__init__()

        # Self Attention layer
        self.self_attn = MultiHeadAttention(num_heads, input_dim)
        # Attention Layer
        self.src_attn = MultiHeadAttention(num_heads, input_dim)

        # Two-layer MLP
        self.linear_net = nn.Sequential(nn.Linear(input_dim, dim_feedforward),
                                        nn.Dropout(dropout),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(dim_feedforward, input_dim))

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):

        # Self-Attention part (use self.norm1)
        # Add your code below

        attn_output= self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(self.norm1(attn_output))

        # Attention part (use self.norm2)
        # Recall that memory is the output of the encoder and replaces x as
        # the key and value in the attention layer
        # Add your code below

        attn_output = self.src_attn(x, memory, memory, src_mask)
        x = x + self.dropout(self.norm2(attn_output))

        # MLP part (use self.norm3)
        # Add your code below

        linear_output = self.linear_net(x)
        x = x + self.dropout(self.norm3(linear_output))

        return x
    

class PositionalEncoding(nn.Module):

    # Implement the PE function.

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)

        # Add your code below

        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)) / d_model) # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pe[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pe[:, 1::2] = torch.cos(positions_list * division_term)

        pe = pe.unsqueeze(0) # the final dimension is (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
    

class Transformer(nn.Module):
    def __init__(self, enc_inp_size, dec_inp_size, dec_out_size, N=6,
                   d_model=512, dim_feedforward=2048, num_heads=8, dropout=0.1,
                   mean=[0,0],std=[0,0]):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.N = N
        self.mean = mean
        self.std = std
        self.enc_inp_size = enc_inp_size
        self.dec_inp_size = dec_inp_size
        self.dec_out_size = dec_out_size

        self.encoder = nn.ModuleList([deepcopy(
            EncoderBlock(d_model, num_heads, dim_feedforward, dropout)) for _ in range(N)])
        self.decoder = nn.ModuleList([deepcopy(
            DecoderBlock(d_model, num_heads, dim_feedforward, dropout)) for _ in range(N)])

        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.pos_dec = PositionalEncoding(d_model, dropout)

        self.src_embed = nn.Linear(enc_inp_size, d_model)
        self.tgt_embed = nn.Linear(dec_inp_size, d_model)

        self.out = nn.Linear(d_model, dec_out_size)

        self.init_weights()


    def forward(self, src, trg, src_mask, trg_mask):

        # First part of the forward pass: embedding and positional encoding
        # both for the source and target

        # Add your code below

        src = self.pos_enc(self.src_embed(src))
        trg = self.pos_enc(self.tgt_embed(trg))

        # Second part of the forward pass: the encoder and decoder layers.
        # Look at the arguments of the forward pass of the encoder and decoder
        # and recall that the encoder output is used as the memory in the decoder.

        # Add your code below

        for layer_idx in range(len(self.encoder)):
          src = self.encoder[layer_idx](src, src_mask)

        for layer_idx in range(len(self.decoder)):
          trg = self.decoder[layer_idx](trg, src, src_mask, trg_mask)

        output = self.out(trg)

        return output

    # Initialize parameters with Glorot / fan_avg.
    def init_weights(self):

        for p in self.encoder.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        for p in self.decoder.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        for p in self.pos_enc.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        for p in self.pos_dec.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        for p in self.src_embed.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        for p in self.tgt_embed.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        for p in self.out.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)


def transformer_inputs(batch, device, inp_frames = 10, inp_enc = 33, inp_dec = 34, ):

    inp = batch[:,:inp_frames, 0:inp_enc].to(device)
    target = batch[:,inp_frames:,inp_enc:].to(device)
    
    zero_tensor = torch.zeros(inp_dec)
    zero_tensor[-1] = 1
    start_of_seq = zero_tensor.unsqueeze(0).unsqueeze(1).repeat(target.shape[0], inp_frames, 1).to(device)
    
    target_c = torch.zeros((target.shape[0], target.shape[1], 1)).to(device)
    target = torch.cat((target, target_c), -1)

    dec_inp = torch.cat((start_of_seq, target), 1)
    src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
    trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
    
    return inp, dec_inp, src_att, trg_att

def subsequent_mask(size):

    # Mask out subsequent positions.

    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(mask) == 0

def train(data_loader,vald_loader, n_epochs, model, scheduler, optimizer, device, path_to_save_model=None):

    train_loss = []
    val_loss = []
    val_loss_best = 1000

    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                        26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                        46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                        75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    
    log_step  = 200
    clip_grad = True
    use_scheduler = True
    save_and_plot = True

    for epoch in range(n_epochs-1):

        running_loss=0
        n=0
        model.train()

        for cnt,batch in enumerate(data_loader):

            batch=batch.float().to(device)
            batch_dim=batch.shape[0]
            n+=batch_dim
                
            my_batch = batch[:,:, dim_used]
            a,b,c,d = transformer_inputs(my_batch, device)
            
            optimizer.zero_grad()
            sequences_predict = tf(a, b, c, d) #.view(-1, output_n, joints_to_consider, 3)
            sequences_predict = sequences_predict[:,:25, :]
            sequences_gt=batch[:, 10:35, dim_used] #.view(-1,output_n,len(dim_used)//3,3)
            loss=mpjpe_error(sequences_predict,sequences_gt)

            if cnt % log_step == 0:
                print('[Epoch: %d, Iteration: %5d]  training loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))

            loss.backward()

            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip_grad)

            optimizer.step()
            running_loss += loss*batch_dim

        train_loss.append(running_loss.detach().cpu()/n)

        model.eval()
        with torch.no_grad():

            running_loss=0
            n=0

            for cnt,batch in enumerate(vald_loader):

                batch=batch.float().to(device)
                batch_dim=batch.shape[0]
                n+=batch_dim
                
                my_batch = batch[:,:, dim_used]
                a,b,c,d = transformer_inputs(my_batch, device)
                sequences_predict = model(a, b, None, None) #.view(-1, output_n, joints_to_consider, 3)
                sequences_predict = sequences_predict[:,:25, :]
                sequences_gt=batch[:, 10:35, dim_used] #.view(-1,output_n,len(dim_used)//3,3)
                    
                loss=mpjpe_error(sequences_predict,sequences_gt)

                if cnt % log_step == 0:
                            print('[Epoch: %d, Iteration: %5d]  validation loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))

                running_loss+=loss*batch_dim

            val_loss.append(running_loss.detach().cpu()/n)

            if running_loss/n < val_loss_best:
                val_loss_best = running_loss/n

        if use_scheduler:
            scheduler.step()

        # save and plot model every 5 epochs
        # Insert your code below. Use the argument path_to_save_model to save the model to the path specified.
        if save_and_plot and epoch in list(range(4, n_epochs, 5)):

            clear_output(wait=True)
            do_my_plot_and_save(tf, train_loss, val_loss, path_to_save_model, "transformer", epoch )

        return val_loss, val_loss_best
        
def do_my_plot_and_save(my_model, train_loss, val_loss, path_to_save_model, model_name, this_epoch):

    #if not exists(path_to_save_model): makedirs(path_to_save_model)
    if not exists(path_to_save_model+ "plots_dir/"): makedirs(path_to_save_model + "plots_dir/")

    torch.save(my_model.state_dict(), path_to_save_model + model_name + "_epoch_"+str(this_epoch+1)+".pt")

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

    plt.savefig(path_to_save_model + "plots_dir/" +"loss_epoch_"+str(this_epoch+1)+".png", bbox_inches='tight')
    plt.show()
