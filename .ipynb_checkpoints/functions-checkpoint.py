import copy
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import numpy as np
from sklearn import preprocessing
import torch.nn as nn
from torch import optim
import time
from tqdm import tqdm
from sklearn.decomposition import PCA

##################################################### DATA PREP #####################################################
def Xy_Timewindow(X, y, hour):
    X_forecast = X.copy()
    for i in range(-hour,hour):
        if i == 0:
            continue
        X_forecast[f"{i}h_00m"] = X[0].shift(i)
        X_forecast[f"{i}h_10m"] = X[1].shift(i)
        X_forecast[f"{i}h_20m"] = X[2].shift(i)
        X_forecast[f"{i}h_30m"] = X[3].shift(i)
        X_forecast[f"{i}h_40m"] = X[4].shift(i)
        X_forecast[f"{i}h_50m"] = X[5].shift(i)
        
    X_forecast.columns = X_forecast.columns.astype(str)
    
    X_forecast = X_forecast.dropna()
    y_forecast, X_forecast = y.align(X_forecast, join="inner")
    return X_forecast, y_forecast

##SPATIAL SLICING

#Return an array of [lat, lon, values for this radar measure]
#This array is order by the radar the closest to "lat" "lon" input
def closest_n_pixelPoint(xarray, lat, lon, n):
    closest = []
    array = xarray.copy()
    array['distance'] = ((xarray.lat - lat)**2 + (xarray.lon - lon)**2)**0.5
    
    # Set the minimum distance to a large value to exclude it
    min_indices = array['distance'].argmin(dim=['lat', 'lon'])
    min_lat = min_indices['lat'].item()
    min_lon = min_indices['lon'].item()
    array['distance'][min_lat,min_lon] = np.nan
    for i in range(n):
        min_indices = array['distance'].argmin(dim=['lat', 'lon'])
        min_lat = min_indices['lat'].item()
        min_lon = min_indices['lon'].item()
        
        closest += [[array['distance'][min_lat,min_lon].lat.values, 
                     array['distance'][min_lat,min_lon].lon.values,
                     array["var0_1_201_surface"][:,min_lat,min_lon].values]]
        array['distance'][min_lat,min_lon] = np.nan
    return closest

def get_neighbor_spatial_timeWindow_df(target_gauge, radar_xarray, coords, rain, radar, n_neigh, time_window, is_plot=False):

    #Creating dataframes of the neighbors
    lat, lon = coords.loc[target_gauge][["lat", "lon"]]
    closest_358 = closest_n_pixelPoint(radar_xarray, lat, lon, n_neigh)
    neighbors = pd.DataFrame()
    neigh_list = []
    for i in range(len(closest_358)):
        #time index like rain
        grouped_data = [closest_358[i][2][j:j+6] for j in range(0, len(closest_358[i][2]), 6)]
        n_i = pd.DataFrame(grouped_data, 
                           index=rain.index,
                           columns=[f"n{i}_00m", f"n{i}_10m", f"n{i}_20m", f"n{i}_30m",f"n{i}_40m", f"n{i}_50m"])

        if (is_plot):
            #Compare with actual radar
            plt.figure(figsize=(15,5))
            plt.title(f"Comparaison of gauje 358 VS {i}")
            plt.plot(rain.index[rain.index > '2010-03-14'], radar[target_gauge].resample("H").mean().loc[rain.index > '2010-03-14'],c="b", label="radar 358", alpha=0.5, linestyle="--")
            plt.plot(rain.index[rain.index > '2010-03-14'], rain[target_gauge].loc[rain.index > '2010-03-14'],c="r", label="Target",alpha=0.5, linestyle=":")
            plt.plot(rain.index[rain.index > '2010-03-14'], n_i.mean(axis=1).loc[rain.index > '2010-03-14'],
                     label=f"closest gauge {i}",
                     c="g", alpha=0.5)
            plt.legend()
            plt.show()
        
        neigh_list += [n_i.copy()]
        
        #Forecast
        for j in range(-time_window,time_window):
            if j == 0:
                continue
            n_i[f"n{i}_{j}h_00m"] = n_i.iloc[:,0].shift(j)
            n_i[f"n{i}_{j}h_10m"] = n_i.iloc[:,1].shift(j)
            n_i[f"n{i}_{j}h_20m"] = n_i.iloc[:,2].shift(j)
            n_i[f"n{i}_{j}h_30m"] = n_i.iloc[:,3].shift(j)
            n_i[f"n{i}_{j}h_40m"] = n_i.iloc[:,4].shift(j)
            n_i[f"n{i}_{j}h_50m"] = n_i.iloc[:,5].shift(j)
        n_i = n_i.dropna()
        n_i = n_i.drop(n_i.index[-1])
        
        #Norm
        scaler = preprocessing.MinMaxScaler()
        n_i[n_i.columns] = scaler.fit_transform(n_i[n_i.columns])
        
        #Concat
        neighbors = pd.concat([neighbors, n_i], axis=1)

    return neighbors
    

####################################################### MODEL ####################################################
class NN_Model(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
            #nn.ReLU(), #Should i put that layer?
        )
        
        
    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return self.layers(xb)


#################################################### TRAINING LOOPS #################################################

def training_loop(model, loss_func, optimizer, X_train, y_train, X_test, y_test, epoch):  
    train_loss_evo = []
    test_loss_evo = []

    for epoch in range(epoch):
        pred = model(X_train)
        loss = loss_func(pred, y_train)
        
        optimizer.zero_grad()
    
        loss.backward()
        
        optimizer.step()
    
        train_loss_evo.append(loss.detach().numpy())
        
        #TESTING
        model.eval()
        with torch.inference_mode(): 
            test_pred = model(X_test)
            test_loss = loss_func(test_pred, y_test)
            
            test_loss_evo.append(test_loss.detach().numpy())
    return train_loss_evo, test_loss_evo

#training loop
def n_batch_training_loop(model, X_train, y_train, X_valid, y_valid, lr_list, weight_decay=0, momentum=0, epoch=100, batch_size=32): 
    train_loss_evo = []
    valid_loss_evo = []
    models = []
    min_losses = []
    
    batch_start = torch.arange(0, len(X_train), batch_size)

    for lr in lr_list:
        start_time = time.time()
        t_l = []
        v_l = []
        model_cpy = copy.deepcopy(model)
        loss_func = nn.MSELoss()
        optimizer = optim.SGD(model_cpy.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        best_model = None
        min_loss = 1000 #Just a huge number
        for epoch in tqdm(range(epoch)):
            model_cpy.train()
            for start in batch_start:
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
            
                pred = model_cpy(X_batch)
                loss = loss_func(pred, y_batch)
                
                optimizer.zero_grad()
            
                loss.backward()
                
                optimizer.step()
            
            t_l.append(loss.detach().numpy())
            
            #VALIDATION
            model_cpy.eval()
            with torch.no_grad(): 
                valid_pred = model_cpy(X_valid)
                valid_loss = loss_func(valid_pred, y_valid)
                v_l.append(valid_loss.detach().numpy())
                if valid_loss.detach().numpy() < min_loss:
                    min_loss = valid_loss.detach().numpy()
                    best_model = copy.deepcopy(model_cpy)
        models.append(best_model)
        min_losses.append(min_loss)
        train_loss_evo.append(t_l)
        valid_loss_evo.append(v_l)
        end_time = time.time()
        print("Time : ", end_time - start_time)
    
    return train_loss_evo, valid_loss_evo, models, min_losses

#training loop
def reduceLR_training_loop(model, X_train, y_train, X_valid, y_valid, lr_list, weight_decay=0, momentum=0, factor=0.1, patience=10, threshold=0.0001 , epoch=100, batch_size=32): 
    train_loss_evo = []
    valid_loss_evo = []
    models = []
    min_losses = []
    
    batch_start = torch.arange(0, len(X_train), batch_size)

    for lr in lr_list:
        t_l = []
        v_l = []
        model_cpy = copy.deepcopy(model)
        loss_func = nn.MSELoss()
        optimizer = optim.SGD(model_cpy.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience, threshold=threshold)
        best_model = None
        for epoch in range(epoch):
            min_loss = 1000 #Just a huge number
            model_cpy.train()
            for start in batch_start:
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
            
                pred = model_cpy(X_batch)
                loss = loss_func(pred, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            t_l.append(loss.detach().numpy())
            
            #VALIDATION
            model_cpy.eval()
            with torch.no_grad(): 
                valid_pred = model_cpy(X_valid)
                valid_loss = loss_func(valid_pred, y_valid)
                v_l.append(valid_loss.detach().numpy())
                if valid_loss.detach().numpy() < min_loss:
                    min_loss = valid_loss.detach().numpy()
                    best_model = copy.deepcopy(model_cpy)
            scheduler.step(valid_loss.item())
        
        models.append(best_model)
        min_losses.append(min_loss)
        train_loss_evo.append(t_l)
        valid_loss_evo.append(v_l)
    
    return train_loss_evo, valid_loss_evo, models, min_losses
    
#training loop with params
def params_training_loop(X_train, y_train, X_valid, y_valid, 
                         l_lr, l_momentum=[0.9], l_velocity=[0.999], l_weight_decay=[0], 
                         epoch=500, batch_size=32): 
    train_loss_evo = []
    valid_loss_evo = []
    models = []
    losses = []
    
    batch_start = torch.arange(0, len(X_train), batch_size)

    hyperparams = np.array(np.meshgrid(l_lr, l_momentum, l_velocity, l_weight_decay)).T.reshape(-1,4)

    for param in tqdm(hyperparams):
        t_l = []
        v_l = []
        model = NN_Model(X_train.shape[1])
        loss_func = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=param[0], betas=(param[1],param[2]), weight_decay=param[3])
        best_model = None
        min_loss = 1000 #Just a huge number
        for e in tqdm(range(epoch)):
            model.train()
            for start in batch_start:
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
            
                pred = model(X_batch)
                loss = loss_func(pred, y_batch)
                
                optimizer.zero_grad()
            
                loss.backward()
                
                optimizer.step()
            
            t_l.append(loss.detach().numpy())
            
            #VALIDATION
            model.eval()
            with torch.no_grad(): 
                valid_pred = model(X_valid)
                valid_loss = loss_func(valid_pred, y_valid)
                v_l.append(valid_loss.detach().numpy())
                if valid_loss.detach().numpy() < min_loss:
                    min_loss = valid_loss.detach().numpy()
                    best_model = copy.deepcopy(model.state_dict())
        models.append(best_model)
        losses.append(min_loss)
        train_loss_evo.append(t_l)
        valid_loss_evo.append(v_l)

    df_params=pd.DataFrame(hyperparams, columns=["lr_list","momentum", "velocity", "weight_decay"])
    return train_loss_evo, valid_loss_evo, models, losses, df_params

################################################ DIMENSION REDUCTION ################################################
def corr_reduction(X, y, threshold, plot=False):
        
    corr = pd.concat([X, y], axis=1).corr()
    target_corr = pd.Series(corr.iloc[:-1,-1], index=corr.index)

    if plot == True:
        plt.figure(figsize=(12,10))
        sns.barplot(x=target_corr, y=target_corr.index)
        plt.show()
    
    #High correlation selection
    target_corr = target_corr[target_corr > 0.4]
    
    Xreturn = X.filter(items=target_corr.index, axis=1)
    
    return Xreturn, y

def pca_reduction(X, n_component, plot=True): #n_component<1 --> percentage pca   
    pca = PCA(n_components=n_component)
    principalComponents = pca.fit_transform(X)
    n=len(pca.explained_variance_ratio_)
    
    #explained variance
    cum_sum = np.cumsum(pca.explained_variance_ratio_)
    if plot==True:
        plt.figure(figsize=(n, 5))
        plt.bar(x=np.arange(1,n+1), height=pca.explained_variance_ratio_)
        plt.step(x=np.arange(1,n+1), y=cum_sum)
        plt.title(f"Cumulative explained variance \nTotal = {cum_sum[-1]}")
        plt.xlabel("component")
        plt.ylabel("explain variance %")
        plt.show()
    
    #Components
    pca_comp = pd.DataFrame(data=pca.components_, columns=X.columns)
    pca_comp = pca_comp.multiply(pca.explained_variance_ratio_, axis=0)

    indices_of_highest_values = pca_comp.sum().nlargest(20).index
    pca_comp = pca_comp[indices_of_highest_values]
    if plot==True:
        plt.figure(figsize=(n, 5))
        pca_comp.plot.bar(stacked=True)
        plt.title("Component composition from dataset most important features")
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.xlabel("component")
        plt.ylabel("composition")
        plt.show()

    return pd.DataFrame(data = principalComponents, index=X.index), pca




############################################# VISUALISATION ################################################

def print_losses(train_loss_evo, valid_loss_evo, min_losses, lr_list, show_all=False):
    #Evolution de la fonction cout
    plt.figure(figsize=(10,5))
    plt.title(f"Loss functions, best loss={min(min_losses)} ")
    plt.scatter(lr_list, min_losses, c="r")
    plt.xlabel("learning rate")
    plt.ylabel("loss")
    plt.show()
    if show_all==False:
        min_index = min_losses.index(min(min_losses))
        #Evolution de la fonction cout
        plt.figure(figsize=(10,5))
        plt.title(f"Train loss and validation loss with learning rate {lr_list[min_index]}")
        plt.plot(valid_loss_evo[min_index], c="r", label="valid")
        plt.plot(train_loss_evo[min_index], label="train")
        plt.legend()
        plt.show()
    else :
        for i in range(len(lr_list)):
            #Evolution de la fonction cout
            plt.figure(figsize=(10,5))
            plt.title(f"Train loss and validation loss with learning rate {lr_list[i]}")
            plt.plot(valid_loss_evo[i], c="r", label="valid")
            plt.plot(train_loss_evo[i], label="train")
            plt.legend()
            plt.show()            

def print_losses_params(train_loss_evo, valid_loss_evo, losses, params, show_all=False):
    #Evolution de la fonction cout
    params["losses"] = losses
    params["losses"] = params["losses"].astype(float)

    plt.figure(figsize=(10,5))
    plt.title(f"Loss functions, best loss={min(losses)} ")
    sns.scatterplot(x="lr_list", y="losses",data=params, hue="momentum", style="weight_decay", size="velocity",
                   sizes=(40, 200), legend="full", palette="deep", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.xlabel("learning rate")
    plt.xscale('log')
    plt.ylabel("loss")
    plt.show()
    
    
    min_index = losses.index(min(losses))
    min_param = params.loc[min_index]
    #Evolution de la fonction cout
    plt.figure(figsize=(10,5))
    plt.title(f"Train loss and validation loss {losses[min_index]} \nlr={params.loc[min_index, 'lr_list']} \nmomentum={params.loc[min_index, 'momentum']} \nveolocity={params.loc[min_index, 'velocity']} \nweight_decay={params.loc[min_index, 'weight_decay']}")
    plt.plot(valid_loss_evo[min_index], c="r", label="valid")
    plt.plot(train_loss_evo[min_index], label="train")
    plt.legend()
    plt.show()
    if show_all==True:
        df=pd.concat([params, pd.Series(train_loss_evo, name="train_loss", dtype=float),  pd.Series(valid_loss_evo,  name="valid_loss", dtype=float)], axis=1)

        len_momentum=len(df["momentum"].unique())
        lenVel=len(df["velocity"].unique())
        lenWD=len(df["weight_decay"].unique())
        
        fig, ax = plt.subplots(len_momentum, lenVel*lenWD, sharex='col', sharey='row', figsize=(10*lenVel*lenWD, 5*len_momentum))
        fig.subplots_adjust(hspace=0.3, wspace=0)
        
        #Fix velocity an dweight decay params
        grouped_df = df.groupby(["velocity", "weight_decay"])
        for col, ((vel, wd), fixed_vel_wd) in enumerate(grouped_df):
            #fix Momentum, plot on the same column
            for row, ((momentum), fixed_mom_vel_wd) in enumerate(fixed_vel_wd.groupby("momentum")):   
                #Different color for the lr_values
                norm = colors.LogNorm(vmin=min(fixed_mom_vel_wd['lr_list']), vmax=max(fixed_mom_vel_wd['lr_list']))
                colormap = plt.cm.RdYlGn
                
                if (len_momentum > 1) & ((lenVel > 1) | (lenWD > 1)):
                    ax[row, col].set_title(f'momentum={momentum}\nvelocity={vel}\nwd={wd}')
                    for index, curve in fixed_mom_vel_wd.iterrows(): 
                        color = colors.to_rgba(colormap(norm(curve["lr_list"])))
                        ax[row, col].plot(curve["train_loss"], color=color, label=f"lr = {curve['lr_list']}", linestyle=":", alpha=0.5)
                        ax[row, col].plot(curve["valid_loss"], color=color,  label=f"lr = {curve['lr_list']}", alpha=0.5)
                    ax[row, col].set_xlabel('Epochs')
                    ax[row, col].set_ylabel('Loss')
                    ax[row, col].legend(loc='upper right')
                elif (len_momentum == 1) & ((lenVel > 1) | (lenWD > 1)):
                    ax[col].set_title(f'momentum={momentum}\nvelocity={vel}\nwd={wd}')
                    for index, curve in fixed_mom_vel_wd.iterrows(): 
                        color = colors.to_rgba(colormap(norm(curve["lr_list"])))
                        ax[col].plot(curve["train_loss"], color=color, label=f"lr = {curve['lr_list']}", linestyle=":", alpha=0.5)
                        ax[col].plot(curve["valid_loss"], color=color,  label=f"lr = {curve['lr_list']}", alpha=0.5)
                    ax[col].set_xlabel('Epochs')
                    ax[col].set_ylabel('Loss')
                    ax[col].legend(loc='upper right')
                elif (lenVel*lenWD == 1) & (len_momentum > 1):
                    ax[row].set_title(f'momentum={momentum}\nvelocity={vel}\nwd={wd}')
                    for index, curve in fixed_mom_vel_wd.iterrows(): 
                        color = colors.to_rgba(colormap(norm(curve["lr_list"])))
                        ax[row].plot(curve["train_loss"], color=color, label=f"lr = {curve['lr_list']}", linestyle=":", alpha=0.5)
                        ax[row].plot(curve["valid_loss"], color=color,  label=f"lr = {curve['lr_list']}", alpha=0.5)
                    ax[row].set_xlabel('Epochs')
                    ax[row].set_ylabel('Loss')
                    ax[row].legend(loc='upper right')                    
                else :
                    ax.set_title(f'momentum={momentum}\nvelocity={vel}\nwd={wd}')
                    for index, curve in fixed_mom_vel_wd.iterrows(): 
                        color = colors.to_rgba(colormap(norm(curve["lr_list"])))
                        
                        ax.plot(curve["train_loss"], color=color, label=f"lr = {curve['lr_list']}", linestyle=":", alpha=0.5)
                        ax.plot(curve["valid_loss"], color=color,  label=f"lr = {curve['lr_list']}", alpha=0.5)
                    ax.set_xlabel('Epochs')
                    ax.set_ylabel('Loss')
                    ax.legend(loc='upper right')
        plt.show()

def plt_squared_error(squared_error):
    squared_error = dict(sorted(squared_error.items(), key=lambda item: item[1]))
    
    # Extract keys and values from the dictionary
    methods = list(squared_error.keys())
    errors = list(squared_error.values())
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(methods, errors, color='skyblue')
    plt.xlabel('Squared Error')
    plt.title('Squared Error by Method')
    plt.gca().invert_yaxis()  # Invert the y-axis to display the highest error at the top
    
    # Show the plot
    plt.show()


