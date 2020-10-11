from rdkit import Chem
import numpy as np
import pandas as pd
import torch


# loading data
def read_sdf(file):
#     file += '_sdf.sdf'
    suppl = Chem.SDMolSupplier(file)
    mol_list = []
    for mol in suppl:
        if mol is None: continue
        mol_list.append(mol)
#     print(file, len(mol_list))
    return mol_list


def _write_sdf(mol_list, writer):
    for mol in mol_list: writer.write(mol)


def mols_to_smiles_list(mols):
    smiles_list = []
    for mol in mols:
        smiles_list.append(Chem.MolToSmiles(mol))
    return smiles_list


def remove_repeated(tobe_deleted, template):
    delete = 0
    length = len(tobe_deleted)
    for i in range(length):
        if tobe_deleted[length - 1 - i] in template:
            tobe_deleted.pop(length - 1 - i)
            delete += 1
    print(delete, 'data have been removed.')
    return tobe_deleted


def smiles_list_to_mols(smi):
    mols = []
    for i in smi:
        mols.append(Chem.MolFromSmiles(i))
    return mols


def smiles_list_to_sdf(smi,route):
    w = Chem.SDWriter(route)
    _write_sdf(smiles_list_to_mols(smi), writer=w)


def get_repeated(tobe_counted, template):
    rep = 0
    replist = []
    length = len(tobe_counted)
    for i in range(length):
        if tobe_counted[i] in template:
            replist.append(i)
            rep += 1
    print(rep, 'data repeated')
    return replist


def get_data_by_index(route, chunksize, index):
    chunk = pd.read_csv(filepath_or_buffer=route,
                        engine='python',
                        iterator=True,
                        header=None,
                        index_col=None)
    try:
        i = 0
        probe = template = 0
        index.sort()
        data = []
        while 1:
            df = chunk.get_chunk(chunksize)
            if index[-1] >= (i+1) * chunksize:
                while index[probe] < (i+1) * chunksize:
                    probe += 1
                batch = index[template:probe]
                template = probe
                i += 1
            else:
                batch = index[template:]
                data.append(df.loc[batch,:])
                raise StopIteration
            # print(batch)
            # print(df)
            # print(i)
            data.append(df.loc[batch,:])
    except StopIteration:
        return pd.concat(data).drop(axis=1, columns=0)


# Molecule Descriptors
class descriptors:
    def smiles_to_ecfp_list(smi_list):
        from rdkit.Chem import rdMolDescriptors
        ecfp = []
        for i in smi_list:
            ecfp.append([int(j) for j in list(rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i),
                                                                       radius=6, nBits = 512).ToBitString())])
        return ecfp

    def smiles_to_descriptors_list(smi_list):
        mol_des = []
        descs = [desc_name[0] for desc_name in Descriptors._descList]
        desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
        for i in smi_list:
            mol_des.append(list(desc_calc.CalcDescriptors(Chem.MolFromSmiles(i))))
        return mol_des

    # Mold2 descriptors

    def clear_NaN(df):
        for columname in df.columns:
            if df[columname].count() != len(df):
                loc = df[columname][df[columname].isnull().values==True].index.tolist()
                print('column: "{}", row {}is NaN'.format(columname,loc))
                # print('列名："{}"'.format(columname))
                df.loc[loc,columname] = 0
        return df

    def drop_NaN(df, col):
        return df.drop(col, axis=1)

    def drop_NaN_cols(df):
        dropped = 0
        for columname in df.columns:
            if df[columname].count() != len(df):
                loc = df[columname][df[columname].isnull().values==True].index.tolist()
                print('column: "{}", row {}is NaN'.format(columname,loc))
                # print('列名："{}"'.format(columname))
                df.drop(columns=columname, inplace=True)
                dropped += 1
        print(dropped, 'columns dropped')
        return df

    def process_Mold2_csv(inr, outr, chunksize, label,
                          remove_repeated=False,
                          drop_col=['D777', 'ReadIn_ID', 'USER_ID']):
        # open to changes
        if remove_repeated == False:
            chunk = pd.read_csv(filepath_or_buffer=inr,
                                engine='python',
                                delim_whitespace=True,
                                iterator=True)
            try:
                while 1:
                # for i in range(1):
                    df = chunk.get_chunk(chunksize)
                    # df = clear_NaN(df)
                    df = drop_NaN(df, col=drop_col)
                    df.loc[:,'label'] = label
                    # print(df)
                    df.to_csv(outr, mode='a',
                              header=False)#index_label, header, columns
            except StopIteration:
                print('finished')
        else:
            chunk = pd.read_csv(filepath_or_buffer=acd_inr,
                                engine='python',
                                delim_whitespace=True,
                                iterator=True)
            try:
                chunkidx = 0
                rep = 0
                while 1:
                    df = chunk.get_chunk(chunksize)
                    for i in range(len(df)):
                        # please write to generate acd_rep
                        if df.loc[i+chunkidx*chunksize,'ReadIn_ID'] in acd_rep:
                            df.drop(i+chunkidx*chunksize, axis=0, inplace=True)
                            rep += 1
                    df = drop_NaN(df, col=drop_col)
                    df.loc[:,'label'] = label
                    df.to_csv(outr, mode='a',
                              header=False)#index_label, header, columns
                    chunkidx += 1
            except StopIteration or ValueError or KeyError:
                print('finished')
                print(rep, 'data repeated')


    def process_mordred_csv(inr, outr, chunksize):

        chunk = pd.read_csv(filepath_or_buffer=inr,
                            engine='python',
                            # header=None,
                            iterator=True)

        try:
            while 1:
                df= chunk.get_chunk(chunksize)
                drop_col = []
                drop_row = []
                for index in df.index:
                    try:
                        float(df.loc[index, 'SpAbs_A'])
                        float(df.loc[index, 'SpDiam_A'])
                    except ValueError:
                        df.drop(index=index, inplace=True)
                        drop_row.append(index)
                for columname in df.columns:
                    try:
                        pd.DataFrame(df.loc[:,columname], dtype=float)
                    except ValueError:
                        # df.drop(columns=columname, inplace=True)
                        drop_col.append(columname)
                        for i in df.index:
                            if type(df.loc[i, columname]) not in [int, float]:
                                df.loc[i, columname] = None

                df.to_csv(outr, index=False, mode='a', header=None)
                print(len(drop_col),'columns changed')
                print(len(drop_row),'rows dropped')
        except StopIteration:
            print('finished')
        # try:
        #     while 1:
        #         df = chunk.get_chunk(chunksize)
        #         probe = dropped = 0
        #         if probe+iteration == 0:
        #             not_float_col = []
        #             for i in range(df.shape[1]):
        #                 try:# columns
        #                     float(df.iloc[0,i])
        #                 except ValueError:
        #                     not_float_col.append(i)
        #         while probe < len(df):
        #             try:
        #                 df.drop(not_float_col, axis=1, inplace=True)
        #                 # print(probe, dropped, iteration)
        #                 row = pd.DataFrame(df.iloc[probe,:], dtype=float)
        #                 probe += 1
        #             except ValueError:
        #                 df.drop(index=iteration*chunksize+probe, inplace=True)
        #                 probe += 1
        #                 dropped += 1
        #         iteration += 1
        #         df.to_csv(outr, index=False, mode='a', header=None)
        # except StopIteration:
        #     print('finished')


    def count_csv_length(route, chunksize):
        chunk = pd.read_csv(filepath_or_buffer=route,
                            engine='python',
                            iterator=True,
                            header=None)
        count = 0
        try:
            while 1:
                count += len(chunk.get_chunk(chunksize))
        except StopIteration:
            print(count, 'data counted')
            return count

    # mordred descriptors

    def mols_to_mordred_csv(mols, path):
        from mordred import Calculator, descriptors
        calc = Calculator(descriptors)
        if len(mols) <= 3000:
            df = calc.pandas(mols)
            df.to_csv(path, index=False)
        else:
            i = 0
            while i*3000 <= len(mols):
                df = calc.pandas(mols[i*3000 : (i+1)*3000])
                df.to_csv(path, index=False, mode='a')
                i+=1
            df = calc.pandas(mols[i*3000 :])
            df.to_csv(path, index=False, mode='a')

    # rdDescriptors

    def generate_rdDescriptors(mol, Normalized=True):
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
        if Normalized:
            generator = rdNormalizedDescriptors.RDKit2DNormalized()
            tors = generator.process(smiles)
        else:
            generator = rdDescriptors.RDKit2D()
            tors = generator.process(smiles)
        return tors[1:]


    def generate_rdDescriptorsSets(mols, Normalized=True):
        from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
        if Normalized:
            generator = rdNormalizedDescriptors.RDKit2DNormalized()
        else:
            generator = rdDescriptors.RDKit2D()

        tors = []
        for mol in mols:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
            tors.append(generator.process(smiles)[1:])

        return np.asarray(tors)


# CNN
class CNN:
    def __init__(self):
        self.chars = \
            ['N','C','I','R','H',
             'O','S','P','V','5'
             '1','2','3','4',')'
             '(','/','-','O','=',
             'N','+','C','\\','#'
             '6','7','8','9','0',
             ']','%','F','L','[',
             'N','s','o','c','n']


    def smiles_process(self, smiles):
        #input a smiles string, return a np.array
        smiles.replace('Cl', 'L')
        smiles.replace('Br', 'R')
    #     smiles.replace('=O', 'Q')
    #     smiles.replace('=N', 'M')
        smiles.replace('Si', 'V')
        return smiles


    def smiles_to_matrix(self, smiles):
        matrix = np.zeros((40,len(smiles)),dtype=int)
        smiles = self.smiles_process(smiles)
        if len(smiles) > 120:
            smiles = smiles[:120]
            matrix = np.zeros((40,120))
        for i in range(len(smiles)):
            if smiles[i] == 'C':
                matrix[1,i] = 1
                matrix[22,i] = 1
            elif smiles[i] == 'N':
                matrix[0,i] = 1
                matrix[20,i] = 1
                matrix[35,i] = 1
            elif smiles[i] == 'O':
                matrix[5,i] = 1
                matrix[18,i] = 1
            else:
                try:
                    matrix[self.chars.index(smiles[i]),i] = 1
                except ValueError:
                    pass

        #padding
        width = matrix.shape[1]
        left = int((120-width)/2)
        right = 120 - width - left
        matrix = np.pad(matrix,((0,0),(left,right)))
        matrix = torch.tensor(matrix, dtype=torch.int8)
        return matrix


    def matrix_save(self, smiles_list):
        matrix_list = torch.zeros([len(smiles_list),40,120], dtype=torch.int8)
        for i in range(len(smiles_list)):
            mol = smiles_list[i]
            matrix = self.smiles_to_matrix(mol)
            matrix_list[i] = matrix
        # matrix_list = torch.tensor(matrix_list, dtype=torch.int8)
        return matrix_list


class ChiralCNN:
    def __init__(self):
        self.chars = \
            ['N','C','I','R','H',
             'O','S','P','V','5'
             '1','2','3','4',')'
             '(','/','-','O','=',
             'N','+','@','\\','#'
             '6','7','8','9','0',
             ']','%','F','L','[',
             '&','s','o','c','n']


    def smiles_process(self, smiles):
        #input a smiles string, return a np.array
        smiles.replace('Cl', 'L')
        smiles.replace('Br', 'R')
        smiles.replace('@@', '&')
    #     smiles.replace('=N', 'M')
        smiles.replace('Si', 'V')
        return smiles


    def smiles_to_matrix(self, smiles):
        matrix = np.zeros((40,len(smiles)),dtype=int)
        smiles = self.smiles_process(smiles)
        if len(smiles) > 260:
            smiles = smiles[:260]
            matrix = np.zeros((40,260))
        for i in range(len(smiles)):
            if smiles[i] == 'N':
                matrix[0,i] = 1
                matrix[20,i] = 1
            elif smiles[i] == 'O':
                matrix[5,i] = 1
                matrix[18,i] = 1
            else:
                try:
                    matrix[self.chars.index(smiles[i]),i] = 1
                except ValueError:
                    pass

        #padding
        width = matrix.shape[1]
        left = int((260-width)/2)
        right = 260 - width - left
        matrix = np.pad(matrix,((0,0),(left,right)))
        matrix = torch.tensor(matrix, dtype=torch.int8)
        return matrix


    def matrix_save(self, smiles_list):
        matrix_list = torch.zeros([len(smiles_list),40,260], dtype=torch.int8)
        for i in range(len(smiles_list)):
            mol = smiles_list[i]
            matrix = self.smiles_to_matrix(mol)
            matrix_list[i] = matrix
        # matrix_list = torch.tensor(matrix_list, dtype=torch.int8)
        return matrix_list
