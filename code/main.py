import numpy
import torch
from utils import set_params, evaluate ,eva
from utils.load_data import *
from module.snhe import SNHE
import warnings
import datetime
import pickle as pkl
import os
import random
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## name of intermediate document ##
own_str = args.dataset

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class_loss = True
multi = True
multi_lenght = 2

def train(a,b):
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test, ae_path = \
        load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]
    print(nb_classes)
    print(feats[0].shape)
    feats_dim_list = [i.shape[1] for i in feats]
    P = int(len(mps))
    print("seed ",args.seed)
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", P)
    
    model = SNHE(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                    P, args.sample_rate, args.nei_num, args.tau, args.lam, nb_classes, ae_path)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        feats = [feat.cuda() for feat in feats]
        mps = [mp.cuda() for mp in mps]
        pos = pos.cuda()
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

    cnt_wait = 0
    best = 1e9
    best_t = 0
    # mps_sum = mps[0]+mps[1]
    # H = torch.mm(mps_sum,mps_sum)
    # mps.append(H)
    # print(mps)
    starttime = datetime.datetime.now()
    with torch.no_grad():
        _, _, _, _, z = model.ae(feats[0])
    z = z.to(device)
    kmeans = KMeans(n_clusters=nb_classes, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    for epoch in range(args.nb_epochs):
        model.train()
        optimiser.zero_grad()
        c_loss,kl_loss,ce_loss = model(feats, pos, mps, nei_index,b)
        embeds,class_loss = model.get_embeds(feats, mps, label, idx_train)
        if class_loss:
            loss = c_loss+a*(kl_loss+ce_loss) + class_loss
        else:
            loss = c_loss+a*(kl_loss+ce_loss)
        print("loss ", loss.data.cpu())
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'SNHE_'+own_str+'.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        loss.backward()
        optimiser.step()
        
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('SNHE_'+own_str+'.pkl'))
    model.eval()
    os.remove('SNHE_'+own_str+'.pkl')
    embeds,_ = model.get_embeds(feats, mps, label, idx_train)
    
    kmeans = KMeans(n_clusters=nb_classes, n_init=20)
    y_pred = kmeans.fit_predict(embeds.data.cpu().numpy())
    eva(torch.argmax(label.cpu(),dim=1), y_pred, 'pae')
    for i in range(len(idx_train)):
        evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                 args.eva_lr, args.eva_wd)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")
    
    if args.save_emb:
        f = open("./embeds/"+args.dataset+"/"+str(args.turn)+".pkl", "wb")
        pkl.dump(embeds.cpu().data.numpy(), f)
        f.close()
#Recommended hyperparameters        
#ACM 0.01 0.2
#DBLP 0.2 0.0005
#IMDB 0.1,0.1
#Aminer 0.1 0.2

if __name__ == '__main__':
    for a in [0.0001,0.0005,0.01,0.05,0.1,0.2,0.5,1]:
        for b in [0.0001,0.0005,0.01,0.05,0.1,0.2,0.5,1]:
    #a=0.1
    #b=0.01
            train(a,b)
