#%%
import pandas as pd
import numpy as np
# Load the CSV file
data = pd.read_csv('data.csv')

# Extract unique CIDs from the "API" column
molecules_API = data["API_Smiles"]
molecules_exp = data["Excipient_Smiles"]

#%%
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors, Descriptors3D
from rdkit.Chem import AllChem

def getMolDescriptors(mol, missingVal=None):
    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res

#%%
API = []
for smiles in molecules_API:
    mol = Chem.MolFromSmiles(smiles)
    des = getMolDescriptors(mol)
    API.append(des)    

#%%
exp = []
for smiles in molecules_exp:
    mol = Chem.MolFromSmiles(smiles)
    des = getMolDescriptors(mol)
    exp.append(des)

#%%
import pandas as pd

# Create DataFrames for the ECFP descriptors
df_API = pd.DataFrame(API)
df_exp = pd.DataFrame(exp)
data_2D = pd.concat([df_API, df_exp, data], axis=1)
print(data_2D.shape)
#%%
# Save the concatenated data to a CSV file
data_2D.to_csv('2D_data.csv', index=False)
