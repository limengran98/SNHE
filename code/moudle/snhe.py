import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .mp_encoder import Mp_encoder
from .sc_encoder import Sc_encoder
from .contrast import Contrast
from torch_geometric.nn import HeteroConv, Linear
from torch_geometric.utils import from_scipy_sparse_matrix,dense_to_sparse
import GCL.losses as L
import GCL.augmentors as A
from GCL.models.contrast_model import WithinEmbedContrast
class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        # extracted feature by AE
        self.z_layer = Linear(n_enc_3, n_z)
        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)
    def forward(self, x):
        enc_z2 = F.relu(self.enc_1(x))
        enc_z3 = F.relu(self.enc_2(enc_z2))
        enc_z4 = F.relu(self.enc_3(enc_z3))
        z = self.z_layer(enc_z4)
        dec_z2 = F.relu(self.dec_1(z))
        dec_z3 = F.relu(self.dec_2(dec_z2))
        dec_z4 = F.relu(self.dec_3(dec_z3))
        x_bar = self.x_bar_layer(dec_z4)

        return x_bar, enc_z2, enc_z3, enc_z4, z

class SNHE(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam, classes, ae_path):
        super(SNHE, self).__init__()
        self.ae = AE(
            n_enc_1=500,
            n_enc_2=500,
            n_enc_3=2000,
            n_dec_1=2000,
            n_dec_2=500,
            n_dec_3=500,
            n_input=feats_dim_list[0],
            n_z=10)
        self.ae.load_state_dict(torch.load( ae_path, map_location='cpu'))   ######
        self.cluster_layer = Parameter(torch.Tensor(classes, 10))  ##########
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.hidden_dim = hidden_dim
        self.classes = classes
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.mp = Mp_encoder(P, hidden_dim, attn_drop)
        self.sc = Sc_encoder(hidden_dim, sample_rate, nei_num, attn_drop)      
        self.contrast = Contrast(hidden_dim, tau, lam)
        self.head = nn.Linear(hidden_dim, classes)    ########
        #self.aug1 = A.MarkovDiffusion()
        self.aug1 = A.FeatureMasking(pf=0.5)
        #self.aug2 = A.NodeDropping(pn=0.5)
        self.aug2 = A.EdgeRemoving(pe=0.5)
        self.contrast_model = WithinEmbedContrast(loss=L.VICReg())#(loss=L.BarlowTwins())
        self.ce = nn.CrossEntropyLoss()

        self.v = 1.0
    def target_distribution(self, q):
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def forward(self, feats, pos, mps, nei_index,b):  # p a s
        x_bar, h1, h2, h3, AE_z = self.ae(feats[0])
        h_all = []
        Mp1 = []
        Mp2 = []
        Mp = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.fc_list[i](feats[i])))
        
        for i in range(len(mps)):
            H1, mp1, _ = self.aug1(h_all[0], dense_to_sparse(mps[i])[0], dense_to_sparse(mps[i])[1])
            H2, mp2, _ = self.aug2(h_all[0], dense_to_sparse(mps[i])[0], dense_to_sparse(mps[i])[1])
            Mp1.append(mp1)
            Mp2.append(mp2)
            Mp.append(dense_to_sparse(mps[i])[0])
        z_mp1 = self.mp(H1, Mp1)
        z_mp2 = self.mp(H2, Mp2)
        #embeds = self.get_embeds(h_all[0], mps)
        z_mp = self.mp(h_all[0], Mp)
        z_sc = self.sc(h_all, nei_index)
        
        GBT_loss = self.contrast_model(z_mp1, z_mp2)
        co_loss = self.contrast(z_mp, z_sc, pos)
        print(GBT_loss)
        print(co_loss)
        #loss = self.contrast(z_mp, z_sc, pos) + 
        loss=  co_loss +  b* GBT_loss


        net_output = self.head(z_mp+z_sc) 
        predict = F.softmax(net_output, dim=1)

        q = 1.0 / (1.0 + torch.sum(torch.pow(AE_z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        tmp_q = q.data
        p = self.target_distribution(tmp_q)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(predict.log(), p, reduction='batchmean')
        print(kl_loss)
        print(ce_loss)
        return loss,kl_loss,ce_loss

    def get_embeds(self, feats, mps, label, idx_train):
        Mp = []
        for i in range(len(mps)):
            Mp.append(dense_to_sparse(mps[i])[0])
        z_mp = F.elu(self.fc_list[0](feats[0]))
        z_mp = self.mp(z_mp, Mp)
        pred = self.head(z_mp) 
        ce_loss = self.ce(pred[idx_train[0]], label[idx_train[0]]) 
        return z_mp.detach(), ce_loss
