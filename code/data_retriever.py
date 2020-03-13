import os
import numpy as np
import pandas as pd
import collections
from torch.utils.data import TensorDataset
from torch import from_numpy
from six import StringIO
from copy import copy
from math import ceil

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_ENTRY = collections.namedtuple("data_entry", "species, mhc, pep_length, sequence, ineq, ic50")

COMMON_AMINO_ACIDS = collections.OrderedDict(sorted({
    "A": "Alanine",
    "R": "Arginine",
    "N": "Asparagine",
    "D": "Aspartic Acid",
    "C": "Cysteine",
    "E": "Glutamic Acid",
    "Q": "Glutamine",
    "G": "Glycine",
    "H": "Histidine",
    "I": "Isoleucine",
    "L": "Leucine",
    "K": "Lysine",
    "M": "Methionine",
    "F": "Phenylalanine",
    "P": "Proline",
    "S": "Serine",
    "T": "Threonine",
    "W": "Tryptophan",
    "Y": "Tyrosine",
    "V": "Valine",
}.items()))
COMMON_AMINO_ACIDS_WITH_UNKNOWN = copy(COMMON_AMINO_ACIDS)
COMMON_AMINO_ACIDS_WITH_UNKNOWN["X"] = "Unknown"

AMINO_ACID_INDEX = dict(
    (letter, i) for (i, letter) in enumerate(COMMON_AMINO_ACIDS_WITH_UNKNOWN))

AMINO_ACIDS = list(COMMON_AMINO_ACIDS_WITH_UNKNOWN.keys())

BLOSUM62_MATRIX = pd.read_csv(StringIO("""
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  X
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3  0
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  0
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  0
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1  0
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  0
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3  0
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3  0
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1  0
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1  0
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1  0
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2  0
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0  0 
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3  0
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1  0
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4  0
X  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1
"""), sep='\s+').loc[AMINO_ACIDS, AMINO_ACIDS]
assert (BLOSUM62_MATRIX == BLOSUM62_MATRIX.T).all().all()

def parse_bdata_input(kmers):
    '''
    Takes an input bdata2013 file and returns a DataFrame, created out of it
    
    Parameters:
    ----------
    kmers: list of ints, representing the lengths of peptides to keep
    
    Returns:
    -------
    pandas.DataFrame
    '''
    filename = os.path.join(BASE_DIR,
                     '..',
                     'data',
                     'bdata.20130222.mhci.txt')
    fullDict = {'species': [], 'mhc': [], 'peptide_length': [], 'sequence': [], 'inequality': [], 'meas': []}
    for line in open(filename, 'r'):
        if ('species' in line):
            pass
        else:
            datalist = line.split()
            if 'HLA' not in datalist[1]:
                continue
            fullDict['species'].append(datalist[0])
            fullDict['mhc'].append(datalist[1])
            fullDict['peptide_length'].append(datalist[2])
            fullDict['sequence'].append(datalist[3])
            fullDict['inequality'].append(datalist[4])
            fullDict['meas'].append(datalist[5])
            
    data = pd.DataFrame.from_dict(fullDict)
    data['peptide_length'] = pd.to_numeric(data['peptide_length'])
    data['meas'] = pd.to_numeric(data['meas'])
    #if kmers not None:
    data = data[data['peptide_length'] == kmers]
    return data    
        
def valid_allele_list():
    l = list()
    f = os.path.join(BASE_DIR,
                     '..',
                     'data',
                     'mhc_i_protein_sequence.txt')
    with open(f, 'r') as in_file:
        for line_num, line in enumerate(in_file):
            if line_num > 0:
                allele = line.strip('\n').split()[0]
                l.append(allele)
    return l

def bdata_HLA_allele_list(length_list):
    data = parse_bdata_input()
    all_allele = list(set([x.mhc for x in data]))
    return list(set(list(filter(lambda x: 'HLA' in x, all_allele))))

def from_ic50(ic50, max_ic50=50000.0):
    """
    Convert ic50s to regression targets in the range [0.0, 1.0].
    
    Parameters
    ----------
    ic50 : numpy.array of float

    Returns
    -------
    numpy.array of float

    """
    x = 1.0 - (numpy.log(ic50) / numpy.log(max_ic50))
    return numpy.minimum(
        1.0,
        numpy.maximum(0.0, x))


def to_ic50(x, max_ic50=50000.0):
    """
    Convert regression targets in the range [0.0, 1.0] to ic50s in the range
    [0, 50000.0].
    
    Parameters
    ----------
    x : numpy.array of float

    Returns
    -------
    numpy.array of float
    """
    return max_ic50 ** (1.0 - x)

def get_hla_aligned_sequence(hla_alle):
    """
    Reading data from [mhc_i_protein_sequence.txt]
    """
    max_length = 0
    f = os.path.join(BASE_DIR,
                     '..',
                     'data',
                     'mhc_i_protein_sequence.txt')
    with open(f, 'r') as in_file:
        for line_num, line in enumerate(in_file):
            if line_num == 0:
                max_length = int(line.strip('\n').split()[1])
            else:
                info = line.strip('\n').split()
                if info[0] == hla_alle:
                    sequence = info[1]
                    break
    return max_length, sequence

def index_encoding(sequences, letter_to_index_dict):
    """
    Encode a sequence of same-length strings to a matrix of integers of the
    same shape. The map from characters to integers is given by
    `letter_to_index_dict`.

    Given a sequence of `n` strings all of length `k`, return a `k * n` array where
    the (`i`, `j`)th element is `letter_to_index_dict[sequence[i][j]]`.

    Parameters
    ----------
    sequences : list of length n of strings of length k
    letter_to_index_dict : dict : string -> int

    Returns
    -------
    numpy.array of integers with shape (`k`, `n`)
    """
    df = pd.DataFrame(iter(s) for s in sequences)
    result = df.replace(letter_to_index_dict)
    return result.values


def fixed_vectors_encoding(index_encoded_sequences, letter_to_vector_df):
    """
    Given a `n` x `k` matrix of integers such as that returned by `index_encoding()` and
    a dataframe mapping each index to an arbitrary vector, return a `n * k * m`
    array where the (`i`, `j`)'th element is `letter_to_vector_df.iloc[sequence[i][j]]`.

    The dataframe index and columns names are ignored here; the indexing is done
    entirely by integer position in the dataframe.

    Parameters
    ----------
    index_encoded_sequences : `n` x `k` array of integers

    letter_to_vector_df : pandas.DataFrame of shape (`alphabet size`, `m`)

    Returns
    -------
    numpy.array of integers with shape (`n`, `k`, `m`)
    """
    (num_sequences, sequence_length) = index_encoded_sequences.shape
    target_shape = (
        num_sequences, sequence_length, letter_to_vector_df.shape[0])
    result = letter_to_vector_df.iloc[
        list(index_encoded_sequences.flatten())
    ].values.reshape(target_shape)
    return result

'''
def get_training_data():
    train_ligands = parse_bdata_input([9])
    allele_list = valid_allele_list()
    
    allele_names = []
    allele_seqs = []
    peptides = []
    targets = []
    #inequalities = []
    for _, entry in train_ligands.iterrows():
        mhc = entry['mhc']
        if mhc not in allele_list:
            continue
            
        _, allele_sequence = get_hla_aligned_sequence(mhc)
        allele_names.append(mhc)
        allele_seqs.append(allele_sequence)
        peptides.append(entry['sequence'])
        targets.append([from_ic50(entry['meas']), 1])
        #inequalities.append(entry['inequality'])
        
    #fixed_length_sequences_al = pd.Series(allele_seqs)
    #fixed_length_sequences_pep = pd.Series(peptides)
    
    index_encoded_alleles = index_encoding(allele_seqs, AMINO_ACID_INDEX)
    vector_encoded_alleles = fixed_vectors_encoding(index_encoded_alleles, BLOSUM62_MATRIX)
    index_encoded_peptides = index_encoding(peptides, AMINO_ACID_INDEX)
    vector_encoded_peptides = fixed_vectors_encoding(index_encoded_peptides, BLOSUM62_MATRIX)
    
    return vector_encoded_alleles, vector_encoded_peptides, np.array(targets)
'''

def random_peptides(num, length=9, distribution=None):
    if num == 0:
        return []
    if distribution is None:
        distribution = pd.Series(
            1, index=sorted(COMMON_AMINO_ACIDS))
        distribution /= distribution.sum()

    return [
        ''.join(peptide_sequence)
        for peptide_sequence in
        np.random.choice(
            distribution.index,
            p=distribution.values,
            size=(int(num), int(length)))
    ]

def split_samples(samples, validate_ratio=0.2):
    for _ in range(10):
        samples.sample(frac=1)
    train_count = ceil(samples.shape[0] * (1 - validate_ratio))
    return samples.iloc[:train_count, :], samples.iloc[train_count:, :]

def make_train_dataset(samples, add_negatives=False):
    """
    Creates a torch Dataset from pandas DataFrame
    
    Parameters
    ----------
    samples : pandas DataFrame containing mhc and peptide data
    
    Returns
    -------
    torch Dataset
    """
    protein_list = []
    features_list = []
    targets_list = []
    random_negative_aff_min=20000.0
    random_negative_aff_max=50000.0
    allele_list = valid_allele_list()
    for _, sample in samples.iterrows():
        mhc = sample['mhc']
        if mhc not in allele_list: 
            continue
            
        _, seq = get_hla_aligned_sequence(mhc)
        index_encoded_allele = index_encoding([seq], AMINO_ACID_INDEX)
        index_encoded_peptide = index_encoding([sample['sequence']], AMINO_ACID_INDEX)
        print(list(index_encoded_peptide.flatten()))
        protein_feature = fixed_vectors_encoding(index_encoded_allele, BLOSUM62_MATRIX)
        ligand_feature = fixed_vectors_encoding(index_encoded_peptide, BLOSUM62_MATRIX)
        if add_negatives:
            protein_list.append(protein_feature)
        features_list.append(np.concatenate(protein_feature, ligand_feature))
        targets_list.append(np.array([from_ic50(sample['meas']), 1]))
        
    if add_negatives:
        N = len(targets_list)
        for i in range(5*N):
            negative_ligand_feature = random_peptides(1, length=9)
            negative_protein_feature = np.random.choice(np.array(protein_list),
                                                         size=negative_ligand_features.shape)
            features_list.append(np.concatenate(negative_protein_feature, negative_ligand_feature))
            targets_list.append(np.concatenate(
                from_ic50(np.random.uniform(
                    random_negative_aff_min, 
                    random_negative_aff_max, 
                    1
                )),
                0
            ))
    features = from_numpy(np.array(features_list))
    targets = from_numpy(np.array(targets_list))
    return TensorDataset(features, targets)

def make_test_dataset(test_file):
    '''
    Parameters:
    ----------
    test_file: path to file of test data
    
    Assume format
    ```
        allele peptide
        ....
    ```
    Returns:
    -------
    torch Tensor
    !!!Only supports 9-length peptides!!!
    '''
    samples = []
    sample_names = []
    with open(test_file, 'r') as in_file:
        for line in in_file:
            info = line.strip('\n').split()
            sample_names.append((info[0], info[1]))
            _, seq = get_hla_aligned_sequence(info[0])
            index_encoded_allele = index_encoding([seq], AMINO_ACID_INDEX)
            index_encoded_peptide = index_encoding([info[1]], AMINO_ACID_INDEX)
            protein_feature = fixed_vectors_encoding(index_encoded_allele, BLOSUM62_MATRIX)
            ligand_feature = fixed_vectors_encoding(index_encoded_peptide, BLOSUM62_MATRIX)
            samples.append(np.concatenate(protein_feature, ligand_feature))

    return from_numpy(np.array(samples)), sample_names

class DataGenerator():
    def __init__(self, batch_size, samples, validate=False):
        self.batch_size = batch_size
        self.samples = samples   # may be changed
        self.allele_list = valid_allele_list()
        self.validate = validate
        self.proteins_features_map = {}
        self.ligand_feature_map = {}

        self.init_data()
        
    def __len__(self):
        return math.ceil(self.samples.shape[0] / self.batch_size)
    
    def init_data(self):
        for sample in self.samples:
            #sample = DATA_ENTRY(*sample)
            mhc = sample['mhc']
            if mhc not in self.allele_list: 
                continue
            
            if mhc not in self.proteins_features_map:
                _, seq = get_hla_aligned_sequence(mhc)
                index_encoded_alleles = index_encoding([seq], AMINO_ACID_INDEX)
                self.proteins_features_map[mhc] = fixed_vectors_encoding(index_encoded_alleles, BLOSUM62_MATRIX)

            if sample['sequence'] not in self.ligand_feature_map:
                index_encoded_peptides = index_encoding(sample['sequence'], AMINO_ACID_INDEX)
                self.ligand_feature_map[sample['sequence']] = fixed_vectors_encoding(index_encoded_peptides, BLOSUM62_MATRIX)

    def __next__(self):
        protein_feature_list = []
        ligand_feature_list = []
        target_values = []
        sampled = pd.DataFrame(columns=['species', 'mhc', 'peptide_length', 'sequence', 'inequality', 'meas'])
        
        i = 0
        while i < self.batch_size:
            sample = self.samples.sample(n=1)
            
            if (sampled == sample.values).all(1).any():
                continue
                
            if sample['mhc'] not in self.allele_list:
                continue
                
            protein_feature_list.append(self.proteins_features_map[sample['mhc']])
            ligand_feature_list.append(self.ligand_feature_map[sample['sequence']])
            target_values.append([from_ic50(sample['meas']), 1])
            
            sampled.append(sample, ignore_index=True)
            i += 1
            
        return (
                    {
                        'protein': np.array(protein_feature_list),
                        'ligand': np.array(ligand_feature_list)
                    },
                    np.array(target_values)
                )
        
        
        
        
        
        