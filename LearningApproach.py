
import random, os, copy, torch, torch.nn as nn, numpy as np, pandas as pd
from sklearn.utils import resample
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
def upsampling(data, target_col_name):
    np.random.seed(10)
    data_copy = copy.deepcopy(data)
    
    classes_up = np.unique(data_copy[target_col_name].values)

    descending_counts_classes = list(data[target_col_name].value_counts().index)
    
    majority = descending_counts_classes.pop(0)

    data_majority = data_copy[data_copy[target_col_name]==majority]
    new_data_upsampled = copy.deepcopy(data_majority)
    for min_class in descending_counts_classes:
      data_minority = data_copy[data_copy[target_col_name]==min_class]
    
    # Upsample minority class
      data_minority_upsampled = resample(data_minority, 
                                  replace=True,     # sample with replacement
                                  n_samples=np.sum(data_copy[target_col_name]==majority),
                                  random_state=123) # reproducible results

      new_data_upsampled = pd.concat([new_data_upsampled, data_minority_upsampled])
    rand = list(range(new_data_upsampled.shape[0]))
    np.random.shuffle(rand)  #shuffle the indices
    new_data_upsampled = new_data_upsampled.iloc[rand] #Re-index the data
    return new_data_upsampled

def conceptsInstances_to_numeric(data, Embeddings, with_stats=True, only_stats=False, alpha=1):
    random.seed(1)
    datapoints = []
    targets = []
    if with_stats and not only_stats:
        for index in range(data.shape[0]):
            no_positive_example = True
            no_negative_example = True
            stats = pd.Series(data.iloc[index]['pos ex stats'][1:-1].split(",")).astype(float).values
            stats = stats/max(1,stats[-1])
            stats = stats[stats>0]
            if len(stats) >= Embeddings.shape[1]:
                stats = list(stats[:Embeddings.shape[1]])
            else:
                stats = list(stats)+[0]*(Embeddings.shape[1]-len(stats))
            pos = list(filter(lambda x: not x in {', ', ''}, data.iloc[index]['positive examples'][2:-2].split("'")))
            if pos:
                no_positive_example = False
                pos = pd.Series(pos).apply(lambda x: pd.Series(list(Embeddings.loc[x]))).values
            neg = list(filter(lambda x: not x in {', ', ''}, data.iloc[index]['negative examples'][2:-2].split("'")))
            #ipdb.set_trace()
            if neg:
                no_negative_example = False
                neg = pd.Series(neg).apply(lambda x: pd.Series(list(random.random()**alpha * Embeddings.loc[x]))).values
            #ipdb.set_trace()
            if no_negative_example: pos = np.vstack([pos, np.array(stats)])
            else: neg = np.vstack([neg, np.array(stats)])
            datapoint = np.vstack([pos,neg]) if not no_positive_example and not no_negative_example else pos if no_negative_example else neg 
            datapoints.append(datapoint[None,:])
            targets.append(data.iloc[index]["concept length"])
    elif only_stats:
        for index in range(data.shape[0]):
            stats = pd.Series(data.iloc[index]['pos ex stats'][1:-1].split(",")).astype(float).values
            stats = stats/max(1,stats[-1])
            datapoints.append(stats[None,:])
            targets.append(data.iloc[index]['concept length'])
    elif not with_stats:
            for index in range(data.shape[0]):
                no_positive_example = True
                no_negative_example = True
                pos = list(filter(lambda x: not x in {', ', ''}, data.iloc[index]['positive examples'][2:-2].split("'")))
                if pos:
                    no_positive_example = False
                    pos = pd.Series(pos).apply(lambda x: Embeddings.loc[x]).values
                neg = list(filter(lambda x: not x in {', ', ''}, data.iloc[index]['negative examples'][2:-2].split("'")))
                if neg:
                    no_negative_example = False
                    neg = pd.Series(neg).apply(random.random()**alpha * Embeddings.loc[x]).values
                datapoint = np.vstack([pos,neg]) if not no_positive_example and not no_negative_example else pos if no_negative_example else neg 
                datapoints.append(datapoint[None,:])
                targets.append(data.iloc[index]["concept length"])
            
    return zip((datapoints, targets))

def load_data(X, y, batch_size=128, shuffle=False):
    if shuffle:
        indices = list(range(len(X)))
        random.shuffle(indices)
        X = torch.FloatTensor(X[indices])
        y = torch.FloatTensor(y[indices])
    else:
        X = torch.FloatTensor(X)
        y = torch.tensor(y)
    if len(y) >= batch_size:
        for i in range(0,len(y)-batch_size+1,batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]
    else:
        yield X, y


class ConceptLengthLearner_LSTM(nn.Module):
    
    def __init__(self, n_hidden=256, n_layers=4, input_size=50, output_size=1,
                               drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.input_size = input_size
        self.name = 'LSTM'
        self.key_args = {'n_layers': n_layers, 'n_hidden': n_hidden, 'input_size':input_size, 'out_size':output_size}
        
        self.lstm = nn.LSTM(input_size, n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Sequential(nn.Linear(n_hidden, 50), nn.BatchNorm1d(50), nn.Linear(50, output_size))
      
    
    def forward(self, x):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
                
        r_output, _ = self.lstm(x)
        
        out = self.dropout(r_output)
        
        # Stack up LSTM outputs using view
        out = out[:,-1,:].contiguous().view(-1, self.n_hidden)
        #print("shape: ", out.shape)
        
        out = self.fc(out)
        return out

class ConceptLengthLearner_GRU(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(ConceptLengthLearner_GRU, self).__init__()
        
        self.name = 'GRU'
        self.key_args = {'hidden_dim': hidden_dim, 'n_layers': n_layers}
        self.n_layers = n_layers
        self.n_hidden = hidden_dim
        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.BatchNorm1d(hidden_dim//2), nn.Linear(hidden_dim//2, output_size))

    def forward(self, x):
        r_out, _ = self.gru(x)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out[:,-1,:].view(-1, self.n_hidden)  
        
        # get final output 
        output = self.fc(r_out)
        
        return output
    

class MLP(nn.Module):
    def __init__(self, inp_dim, dropout=0.3, n_layers=100, num_units=50, out_dim=2, seed=1):
        super(MLP, self).__init__()
        linears = []
        self.n_layers = n_layers
        self.name = 'MLP'
        self.key_args = {'n_layers': n_layers, 'n_units':num_units, 'random seed': seed}
        # number of units for each layer
        np.random.seed(seed)
        layer_dims = [inp_dim] + [num_units//np.random.choice(range(1,10)) for _ in range(n_layers-1)]+[out_dim]
        self.layer_dims = layer_dims
        self.__architecture__()
        for i in range(n_layers):
            if i == 0:
                linears.extend([nn.Linear(layer_dims[i], layer_dims[i+1]), nn.ReLU()])
            elif i < n_layers-1 and i%2==0:
                linears.extend([nn.Linear(layer_dims[i], layer_dims[i+1]), nn.BatchNorm1d(layer_dims[i+1]), nn.Tanh()])
            elif i < n_layers-1 and i%2==1:
                linears.extend([nn.Linear(layer_dims[i], layer_dims[i+1]), nn.Dropout(dropout), nn.ReLU()])
            elif i== n_layers-1:
                linears.extend([nn.Linear(layer_dims[i], layer_dims[i+1]), nn.BatchNorm1d(layer_dims[i+1])])
            else:
                linears.extend([nn.Linear(layer_dims[i], layer_dims[i+1])])
        self.linears = nn.ModuleList(linears)
      
    def forward(self, x):
      for l in self.linears:
        x = x.view(x.shape[0], -1)
        x = l(x)
      return x
    def __architecture__(self):
      print("Model architecture:")
      print(self.layer_dims)

class CNN(nn.Module):
    def __init__(self, inp_width=50, inp_height=501, kernel_w=5, kernel_h=20, stride_w=1, stride_h=3, out_size=4, dropout=0.1):
        super(CNN, self).__init__()
        self.name = 'CNN'
        self.key_args = {'in_height': inp_height, 'inp_width': inp_width, 'kernel_h':kernel_h, 'kernel_w':kernel_w, 'out_size': out_size}
        self.conv2d = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(kernel_h,kernel_w), stride=(stride_h,stride_w), padding=(0,0)),
                                    nn.Dropout2d(dropout),
        nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(kernel_h-2,kernel_w-2), stride=(stride_h,stride_w), padding=(0,0)))
        conv_out = 1292
        self.fc = nn.Sequential(nn.Linear(in_features=conv_out, out_features=50), nn.BatchNorm1d(50), nn.Linear(50, out_size))
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv2d(x)
        x = x.view(x.shape[0], -1)
        #print("shape", x.shape)
        x = self.fc(x)
        return x

def random_model(d_test):
  y_pred = []
  for i in range(d_test.shape[0]):
    y_pred.append(random.choice(range(1,1+max(d_test['concept length']))))
  return y_pred

def train(Net, data_X, data_y, data_dir, train_on_gpu=False, epochs=20, batch_size=10, lr=0.005, print_every=20, kf_n_splits=10):
    
    from sklearn.model_selection import KFold
    
    Kf = KFold(n_splits=kf_n_splits)
    print("-------------------- {} starts training --------------------".format(Net.name))
    counter = 0
    best_val_error = 50.
    fold = 0
    All_losses = defaultdict(lambda: [])
    All_acc = defaultdict(lambda: [])
    for train_index, test_index in Kf.split(data_X):
      net = copy.deepcopy(Net)
      opt = torch.optim.SGD(net.parameters(), lr=lr)
      criterion = nn.CrossEntropyLoss()
      
      if(train_on_gpu):
          net.cuda()
      X_train, X_test = data_X[train_index], data_X[test_index]
      y_train, y_test = data_y[train_index], data_y[test_index]
      fold += 1
      print()
      print("*"*50)
      print("Fold {}/{}:\n".format(fold, kf_n_splits))
      print("*"*50, "\n")
      counter = 0
      train_losses = []
      Train_losses = []
      Val_losses = []

      Train_acc = []
      Val_acc = []
      tr_correct_pred = 0.
      tr_total_dpoints_before_eval = 0.
      for e in range(epochs):
          for x, y in load_data(X=X_train, y=y_train, batch_size=batch_size, shuffle=True):
              counter += 1
              if net.name == 'MLP': x = x.mean(dim=1)
              y = y.to(torch.long)
              if(train_on_gpu):
                  x, y = x.cuda(), y.cuda()

              net.zero_grad()
              output = net(x)
              tr_total_dpoints_before_eval += len(y)
              tr_correct_pred += (y==output.argmax(1)).detach().cpu().sum().item()
              # calculate the loss and perform backprop
              loss = criterion(output, y)
              train_losses.append(loss.item())
              loss.backward()
              opt.step()
              
              # loss stats
              if counter % print_every == 0:
                  tr_acc = 100*(tr_correct_pred/tr_total_dpoints_before_eval)
                  # Get validation loss
                  val_losses = []
                  net.eval()
                  correct_pred = 0.
                  for x, y in load_data(X=X_test, y=y_test, batch_size=500, shuffle=False):
                      if net.name == 'MLP': x = x.mean(dim=1)
                      if(train_on_gpu):
                          x, y = x.cuda(), y.cuda()
                      output = net(x)
                      correct_pred += (y==output.argmax(1)).detach().cpu().sum().item()
                      val_loss = criterion(output, y)
                      val_losses.append(val_loss.item())
                  
                  net.train() # reset to train mode after iterationg through validation data
                  Train_losses.append(np.mean(train_losses))
                  Val_losses.append(np.mean(val_losses))
                  val_acc = 100.*(correct_pred/X_test.shape[0])
                  Val_acc.append(val_acc)
                  Train_acc.append(tr_acc)
                  print("Epoch: {}/{}...".format(e+1, epochs),
                        "Step: {}...".format(counter),
                        "Train loss: {:.4f}...".format(np.mean(train_losses)),
                        "Val loss: {:.4f}...".format(np.mean(val_losses)),
                        "Train acc: {:.2f}%...".format(tr_acc),
                        "Val acc: {:.2f}%".format(val_acc))
                  train_losses = []
                  tr_total_dpoints_before_eval = 0.
                  tr_correct_pred = 0.

      weights = copy.deepcopy(net.state_dict())
      if np.mean(Val_losses) < best_val_error:
        best_val_error = np.mean(Val_losses)
        best_weights = weights
      All_losses["train"].append(Train_losses)
      All_losses["val"].append(Val_losses)

      All_acc["train"].append(Train_acc)
      All_acc["val"].append(Val_acc)
    min_num_steps = min(min([len(l) for l in All_losses['train']]), min([len(l) for l in All_losses['val']]))
    train_l = np.array([l[:min_num_steps] for l in All_losses["train"]]).mean(0)
    val_l = np.array([l[:min_num_steps] for l in All_losses["val"]]).mean(0)

    t_acc = np.array([l[:min_num_steps] for l in All_acc["train"]]).mean(0)
    v_acc = np.array([l[:min_num_steps] for l in All_acc["val"]]).mean(0)
    del All_losses, All_acc        
    _, ax = plt.subplots()
    ax.plot(range(len(train_l)), train_l, 'go-', label='Average training loss')
    ax.plot(range(len(val_l)), val_l, 'ro--', label='Average validation loss')
    legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('C0')
    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.show()

    _, Ax = plt.subplots()
    Ax.plot(range(len(t_acc)), t_acc, 'go-', label='Average training accuracy')
    Ax.plot(range(len(v_acc)), v_acc, 'ro--', label='Average validation accuracy')
    Legend = Ax.legend(loc='best', shadow=True, fontsize='x-large')
    Legend.get_frame().set_facecolor('C0')
    plt.xlabel("training steps")
    plt.ylabel("accuracy")
    plt.show()

    net.load_state_dict(best_weights)
    ans = input("Would you like to save the model ? n (no) y (yes)")
    if ans == 'y':
      torch.save(net.state_dict(), "./"+("/").join(data_dir.split("/")[1:-2])+"/"+str(net.name)+"_"+str(net.key_args)+"_.pt")
      print("Model saved")
    print("Train average accuracy: {}, Validation average accuracy: {}".format(t_acc[-1], v_acc[-1]))
    net.eval()
    return net
                
