import time
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.load_data import load_data
from utils.tools import EarlyStopping, index_generator, parse_minibatch_DTI
from model.DTI_lp import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# hyperparams
num_ntype = 4   
dropout_rate = 0.2  
lr = 1e-5   
weight_decay = 0.0005

num_drug = 708
num_protein = 1512

#  0:drug-protein  1:protein-drug  2:drug-disease
#  3:disease-drug  4:drug-se  5:se-drug  6:pro-disease  7:disease-pro
etypes_lists = [[[None], [0, 1], [2, 3], [4, 5]], 
                [[None], [1, 0], [6, 7]]]            

use_masks = [[False, True, False, False],
             [False, True, False]]

no_masks = [[False, False, False, False],
             [False, False, False]]

expected_metapaths = [
    [(0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)],
    [(1, 1), (1, 0, 1), (1, 2, 1)]
]

def run_DTI(feats_type, hidden_dim, num_heads, attn_vec_dim, aggregator,
                     num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix):
    
    adjlists_dt, edge_metapath_indices_list_dt, type_mask, train_val_test_pos_drug_protein, train_val_test_neg_drug_protein = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features_list = []
    in_dims = []
    if feats_type == 0:
        for i in range(num_ntype):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))
    elif feats_type == 1:
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(device))
    train_pos_drug_protein = train_val_test_pos_drug_protein['train_pos_drug_protein']
    val_pos_drug_protein = train_val_test_pos_drug_protein['val_pos_drug_protein']
    test_pos_drug_protein = train_val_test_pos_drug_protein['test_pos_drug_protein']
    train_neg_drug_protein = train_val_test_neg_drug_protein['train_neg_drug_protein']
    val_neg_drug_protein = train_val_test_neg_drug_protein['val_neg_drug_protein']
    test_neg_drug_protein = train_val_test_neg_drug_protein['test_neg_drug_protein']
    y_true_test = np.array([1] * len(test_pos_drug_protein) + [0] * len(test_neg_drug_protein))

    auc_list = []
    ap_list = []
    for _ in range(repeat):

        net = DTI_lp([4, 3], 8, etypes_lists, in_dims, hidden_dim, hidden_dim, num_heads, attn_vec_dim, aggregator, dropout_rate)

        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos_drug_protein))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos_drug_protein), shuffle=False)
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            net.train()
            for iteration in range(train_pos_idx_generator.num_iterations()):
                # forward
                t0 = time.time()

                train_pos_idx_batch = train_pos_idx_generator.next()
                train_pos_idx_batch.sort()
                train_pos_drug_protein_batch = train_pos_drug_protein[train_pos_idx_batch].tolist()
                train_neg_idx_batch = np.random.choice(len(train_neg_drug_protein), len(train_pos_idx_batch))
                train_neg_idx_batch.sort()
                train_neg_drug_protein_batch = train_neg_drug_protein[train_neg_idx_batch].tolist()

                train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists = parse_minibatch_DTI(
                    adjlists_dt, edge_metapath_indices_list_dt, train_pos_drug_protein_batch, device, neighbor_samples, use_masks, num_drug)
                train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists = parse_minibatch_DTI(
                    adjlists_dt, edge_metapath_indices_list_dt, train_neg_drug_protein_batch, device, neighbor_samples, no_masks, num_drug)

                t1 = time.time()
                dur1.append(t1 - t0)

                [pos_embedding_drug, pos_embedding_protein], _ = net(
                    (train_pos_g_lists, features_list, type_mask, train_pos_indices_lists, train_pos_idx_batch_mapped_lists))
                [neg_embedding_drug, neg_embedding_protein], _ = net(
                    (train_neg_g_lists, features_list, type_mask, train_neg_indices_lists, train_neg_idx_batch_mapped_lists))
                pos_embedding_drug = pos_embedding_drug.view(-1, 1, pos_embedding_drug.shape[1])
                pos_embedding_protein = pos_embedding_protein.view(-1, pos_embedding_protein.shape[1], 1)
                neg_embedding_drug = neg_embedding_drug.view(-1, 1, neg_embedding_drug.shape[1])
                neg_embedding_protein = neg_embedding_protein.view(-1, neg_embedding_protein.shape[1], 1)
                pos_out = torch.bmm(pos_embedding_drug, pos_embedding_protein)
                neg_out = -torch.bmm(neg_embedding_drug, neg_embedding_protein)
                train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

                t2 = time.time()
                dur2.append(t2 - t1)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info
                if iteration % 20 == 0:
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
            # validation
            net.eval()
            val_loss = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_pos_drug_protein_batch = val_pos_drug_protein[val_idx_batch].tolist()
                    val_neg_drug_protein_batch = val_neg_drug_protein[val_idx_batch].tolist()
                    val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists = parse_minibatch_DTI(
                        adjlists_dt, edge_metapath_indices_list_dt, val_pos_drug_protein_batch, device, neighbor_samples, no_masks, num_drug)
                    val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists = parse_minibatch_DTI(
                        adjlists_dt, edge_metapath_indices_list_dt, val_neg_drug_protein_batch, device, neighbor_samples, no_masks, num_drug)

                    [pos_embedding_drug, pos_embedding_protein], _ = net(
                        (val_pos_g_lists, features_list, type_mask, val_pos_indices_lists, val_pos_idx_batch_mapped_lists))
                    [neg_embedding_drug, neg_embedding_protein], _ = net(
                        (val_neg_g_lists, features_list, type_mask, val_neg_indices_lists, val_neg_idx_batch_mapped_lists))
                    
                    pos_embedding_drug = pos_embedding_drug.view(-1, 1, pos_embedding_drug.shape[1])
                    pos_embedding_protein = pos_embedding_protein.view(-1, pos_embedding_protein.shape[1], 1)
                    neg_embedding_drug = neg_embedding_drug.view(-1, 1, neg_embedding_drug.shape[1])
                    neg_embedding_protein = neg_embedding_protein.view(-1, neg_embedding_protein.shape[1], 1)

                    pos_out = torch.bmm(pos_embedding_drug, pos_embedding_protein)
                    neg_out = -torch.bmm(neg_embedding_drug, neg_embedding_protein)

                    val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)))
                val_loss = torch.mean(torch.tensor(val_loss))
            t_end = time.time()

            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_pos_drug_protein), shuffle=False)
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        pos_proba_list = []
        neg_proba_list = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_pos_drug_protein_batch = test_pos_drug_protein[test_idx_batch].tolist()
                test_neg_drug_protein_batch = test_neg_drug_protein[test_idx_batch].tolist()
                test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_minibatch_DTI(
                    adjlists_dt, edge_metapath_indices_list_dt, test_pos_drug_protein_batch, device, neighbor_samples, no_masks, num_drug)
                test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_minibatch_DTI(
                    adjlists_dt, edge_metapath_indices_list_dt, test_neg_drug_protein_batch, device, neighbor_samples, no_masks, num_drug)

                [pos_embedding_drug, pos_embedding_protein], _ = net(
                    (test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists))
                [neg_embedding_drug, neg_embedding_protein], _ = net(
                    (test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists))
                with open('./pos_embedding_drug.txt','a+') as f1:
                    np.savetxt(f1, pos_embedding_drug.cpu())
                with open('./pos_embedding_protein.txt','a+') as f2:
                    np.savetxt(f2, pos_embedding_protein.cpu())
                with open('./neg_embedding_drug.txt','a+') as f3:
                    np.savetxt(f3, neg_embedding_drug.cpu())
                with open('./neg_embedding_protein.txt','a+') as f4:
                    np.savetxt(f4, neg_embedding_protein.cpu())
                
                pos_embedding_drug = pos_embedding_drug.view(-1, 1, pos_embedding_drug.shape[1])
                pos_embedding_protein = pos_embedding_protein.view(-1, pos_embedding_protein.shape[1], 1)
                neg_embedding_drug = neg_embedding_drug.view(-1, 1, neg_embedding_drug.shape[1])
                neg_embedding_protein = neg_embedding_protein.view(-1, neg_embedding_protein.shape[1], 1)

                pos_out = torch.bmm(pos_embedding_drug, pos_embedding_protein).flatten()
                neg_out = torch.bmm(neg_embedding_drug, neg_embedding_protein).flatten()
                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))
            
            
            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().numpy()
        auc = roc_auc_score(y_true_test, y_proba_test)
        ap = average_precision_score(y_true_test, y_proba_test)
        print('DTI PREDICTION TEST:')
        print('AUROC = {}'.format(auc))
        print('AUPR = {}'.format(ap))
        auc_list.append(auc)
        ap_list.append(ap)

    print('----------------------------------------------------------------')
    print('DTI PREDICTION TEST RESULTS:')
    print('AUROC_mean = {}'.format(np.mean(auc_list)))
    print('AUPR_mean = {}'.format(np.mean(ap_list)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MHTAN-DTI')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - all zero vector. Default is 0.')
    ap.add_argument('--hidden-dim', type=int, default=128, help='Dimension of the node hidden state. Default is 128.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--aggregator', default='transformer', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=120, help='Number of epochs. Default is 120.')
    ap.add_argument('--patience', type=int, default=120, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=32, help='Batch size. Default is 32.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='DTI', help='Postfix for the saved model and result. Default is DTI.')

    args = ap.parse_args()
    run_DTI(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.aggregator, args.epoch,
                     args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix)
                   
