import numpy as np
import pickle

def load_data(prefix='DTI_data'):
    in_file = open(prefix + '/0/0-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()
    in_file = open(prefix + '/0/0-2-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02
    in_file.close()
    in_file = open(prefix + '/0/0-3-0.adjlist', 'r')
    adjlist03 = [line.strip() for line in in_file]
    adjlist03 = adjlist03
    in_file.close()
    in_file = open(prefix + '/1/1-1.adjlist', 'r')
    adjlist10 = [line.strip() for line in in_file]
    adjlist10 = adjlist10
    in_file.close()
    in_file = open(prefix + '/1/1-0-1.adjlist', 'r')
    adjlist11 = [line.strip() for line in in_file]
    adjlist11 = adjlist11
    in_file.close()
    in_file = open(prefix + '/1/1-2-1.adjlist', 'r')
    adjlist12 = [line.strip() for line in in_file]
    adjlist12 = adjlist12
    in_file.close()
    
    in_file = open(prefix + '/0/0-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-2-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-3-0_idx.pickle', 'rb')
    idx03 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-1_idx.pickle', 'rb')
    idx10 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-1_idx.pickle', 'rb')
    idx11 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-2-1_idx.pickle', 'rb')
    idx12 = pickle.load(in_file)
    in_file.close()

    # adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    train_val_test_pos_drug_protein = np.load(prefix + '/train_val_test_pos_drug_protein.npz')
    train_val_test_neg_drug_protein = np.load(prefix + '/train_val_test_neg_drug_protein.npz')

    return [[adjlist00, adjlist01, adjlist02, adjlist03],[adjlist10, adjlist11, adjlist12]],\
           [[idx00, idx01, idx02, idx03], [idx10, idx11, idx12]],\
           type_mask, train_val_test_pos_drug_protein, train_val_test_neg_drug_protein
