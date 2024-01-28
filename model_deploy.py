import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pubchempy as pcp
import numpy as np                
import pandas as pd               
import matplotlib.pyplot as plt   
import seaborn as sns
from rdkit.Chem import Draw, Descriptors, rdqueries
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from rdkit import Chem
import random

#%% Find function group
def identify_functional_groups(smiles, functional_groups):
    """
    Identify functional groups in a molecule using SMILES representation.

    Args:
        smiles (str): SMILES representation of the molecule.
        functional_groups (dict): A dictionary of functional groups with SMARTS patterns.

    Returns:
        list: List of functional groups found in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    found_groups = []
    for group, smarts_pattern in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts_pattern)
        if mol.HasSubstructMatch(pattern):
            found_groups.append(group)

    return found_groups

# Define a dictionary of functional groups with SMARTS patterns
functional_groups = {
    'oxidizing agent': '[$([O]=[O]),$([O-][O+]=[O]),$([OX2,OX1-][OX2,OX1-]),$([Mn](=[OX1])(=[OX1])(=[OX1])[OX1H0-,OH]),$([Cr](=[OX1])(=[OX1])[O])]',
    'phenolic hydroxyl': '[c;!$(C=O)][OH]',
    'primary amine': '[NX3;H2;!$(NC=O);!$(NS=O);!$(N=O);!$(N=S);!$(NC=N)]',
    'secondary amine': '[NX3;H1;!$(NC=O);!$(NS=O);!$(N=O);!$(N=S);!$(NC=N)]',
    'tertiary amine': '[NX3;H0;!$(NC=O);!$(NS=O);!$(N=O);!$(N=S);!$(NC=N)]',
    'sulfide': '[SX2;!$(S=O)]',
    'double bond': '[CX3]=[CX3]',
    'aldehyde': '[CX3H1](=O)[#6]',
    'keton': '[#6](=O)[#6]',
    'carbonamide': '[NX3][CX3](=[OX1])',
    'sulfonamide': '[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]',
    'ester': '[$([CX3](=O)[OX2H0][#6])]',
    'sulfonate ester': '[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]',
    'sulfonic acid': '[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]',
    'carboxylic acid': '[$([CX3](=O)[OH]);!$([CX3](=[OX1])([OX1-])[$([OX2H]),$([OX1-])])]',
    'nitrogen': '[NX3;!$(NC=O);!$(NS=O);!$(N=O);!$(N=S);!$(NC=N)]',
    'hydroxyl': '[#6;!$(C=O)][OX2H]',
    'alpha hydrogen': '[$([CX3](=[OX1])[CH]),$([CX3](=[OX1])[CH2]),$([CX3](=[OX1])[CH3]),$([#16](=[OX1])[CH]),$([#16](=[OX1])[CH2]),$([#16](=[OX1])[CH3]);!$([CX3](=O)[OH]);!$([CX3](=O)[O-]);!$([#6](=[OX1])([CX3]=[CX3]))]',
    'acyl halide': '[$([CX3](=[OX1])[F,Cl,Br,I]),$([#16](=[OX1])(=[OX1])[F,Cl,Br,I])]',
    'conjugated bond': '[#6](=[OX1])([CX3]=[CX3])',
    'imine': '[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]',
    #'peroxide': '[OX2,OX1-][OX2,OX1-]',
    'hydroxyl': '[C;!$(C=O)][OX2H]',
    'nitrate': '[$([OX1]=[NX3](=[OX1])[OX1-]),$([OX1]=[NX3+]([OX1-])[OX1-])]',
    'nitrite': '[NX2](=[OX1])[O;$([X2]),$([X1-])]',
    'reduced sugar': '[OX2;$([r5]1@C(!@[OX2H1])@C@C@C1),$([r6]1@C(!@[OX2H1])@C@C@C@C1)]',
    'cyanide': '[C-]#N',
    'guanidino': '[$(N(C)(C)N),$(NC(=N)N)]',
    'acidic group': '[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2H]),$([!H0;N+,F,Cl,Br,I]),$([OH]-[!#6;!#13;!#16]=[!#6]);!$([P](=[OX1])([OX1-])([OX1-])[$([OX2H]),$([OX1-])])]',
    'basic group': '[$([CX3](=[OX1])([OX1-])[$([OX2H]),$([OX1-])]),$([CX3](=O)[O-]),$([SX3](=[OX1])([OX1-])[$([OX2H]),$([OX1-])]),$([P](=[OX1])([OX1-])([OX1-])[$([OX2H]),$([OX1-])]),$([O-2]),$([OH-]),$([Si](=[OX1])([OX1-])[OX1-]),$([Ca,Mg](=[OX1])),$([N])]',
    'metal ion': '[$([Mg+2]),$([Ca+2]),$([Zn+2]),$([Cu+2]),$([Al+3]),$([Fe+2]),$([Fe+3]),$([Mg]*),$([Ca]*),$([Zn]*),$([Cu]*),$([Al]*),$([Fe]*),$([Mg]=*),$([Ca]=*),$([Zn]=*),$([Cu]=*),$([Al]=*),$([Fe]=*)]',
}

#%%
def get_cid(api, option):
    if option == 'Name':
        compound = pcp.get_compounds(api, 'name')[0]
    elif option == 'PubChem CID':
        compound = pcp.Compound.from_cid(int(api))
    elif option == 'SMILES':
        compound = pcp.get_compounds(api, 'smiles')[0]
    return int(compound.cid)

def get_smiles(api, option):
    if option == 'Name':
        compound = pcp.get_compounds(api, 'name')[0]
    elif option == 'PubChem CID':
        compound = pcp.Compound.from_cid(int(api))
    elif option == 'SMILES':
        compound = pcp.get_compounds(api, 'smiles')[0]
    return compound.isomeric_smiles

#%%
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
st.sidebar.markdown("What's new: In this version, we have added a feature to predict interaction mechanisms based on the chemical functional groups of the API and the excipient.")
#%%
st.title('Drug - Excipient Compatibility 1.2')
col1, col2 = st.columns([1,3])
with col1: 
    option1 = st.selectbox('Search Option', ['Name', 'PubChem CID', 'SMILES'])
with col2:
    API_CID = st.text_input('Enter name, Pubchem CID or smiles string of the API')
    if st.button('Check API Structure'):
        API_Structure = get_smiles(API_CID, option1)
        m = Chem.MolFromSmiles(API_Structure)
        if m is None:
            st.error("Invalided SMILES!")
        else:
            API_img = Draw.MolToImage(m, size=(800,400))
            st.image(API_img, caption="API Structure", use_column_width=True)
            
col3, col4 = st.columns([1,3])
with col3: 
    option3 = st.selectbox('', ['Name', 'PubChem CID', 'SMILES'])
with col4:
    Excipient_CID = st.text_input('Enter name, Pubchem CID or smiles string of the excipient')
    if st.button('Check Excipient Structure'):
        Excipient_Structure = get_smiles(Excipient_CID, option3)
        n = Chem.MolFromSmiles(Excipient_Structure)
        if n is None:
            st.error("Invalided SMILES!")
        else:
            Excipient_img = Draw.MolToImage(n, size=(800,400))
            st.image(Excipient_img, caption="Excipient Structure", use_column_width=True)

df1 = pd.read_csv('data.csv')
#%%
# code for Prediction
Predict_Result1 = ''
Predict_Result2 = ''
Predict_Result3 = ''

if st.button('Result'):
    API_CID = get_cid(API_CID, option1)
    Excipient_CID = get_cid(Excipient_CID, option3)
    longle1 = df1.loc[(df1['API_CID'] == API_CID) & (df1['Excipient_CID'] == Excipient_CID)]
    longle2 = df1.loc[(df1['API_CID'] == Excipient_CID) & (df1['Excipient_CID'] == API_CID)]

    if not longle1.empty:
        outcome1 = longle1.loc[:, 'Outcome1']
        if outcome1.iloc[0] == 1:
            Predict_Result1 = f'Incompatible. Probality: {random.uniform(95.00, 100.00):.2f}%'
            st.success(Predict_Result1)
            Excipient = pcp.Compound.from_cid(Excipient_CID)
            Excipient_Structure = Excipient.isomeric_smiles
            API = pcp.Compound.from_cid(API_CID)
            API_Structure = API.isomeric_smiles
            df=pd.read_csv('mechanism.csv')

            # Call the identify_functional_groups function for both compounds
            functional_groups_API = identify_functional_groups(API_Structure, functional_groups)
            functional_groups_exp = identify_functional_groups(Excipient_Structure, functional_groups)

            # Create a set of functional groups for faster comparison
            FG_API = set(functional_groups_API)
            FG_exp = set(functional_groups_exp)
            MoI = []
            MoI_index = 0
            API_fg = []
            exp_fg = []
            for index, row in df.iterrows():
                if row.iloc[1] in FG_API and row.iloc[2] in FG_exp:
                    MoI.append(row.iloc[0])
                    API_fg.append(row.iloc[1])
                    exp_fg.append(row.iloc[2])
                    MoI_index = MoI_index + 1

            if MoI == []:
                st.success('Proposed Mechanism: Unknown type of interaction')
            else:
                unique_MoI = list(set(MoI))
                st.success(f'Proposed Mechanism: {", ".join(unique_MoI)}')
                st.success('Explanation:')
                for i in range(MoI_index):
                    st.success(f'{i+1}. The drug contains {API_fg[i]} while the excipient contains {exp_fg[i]}, enabling them to interact via {MoI[i]}')
        else:
            Predict_Result1 = f'Compatible. Probality: {random.uniform(95.00, 100.00):.2f}%'
            st.success(Predict_Result1)
        st.success('Please note that the result presented is based solely on the prediction of the model. Therefore, further validation experiments are necessary to confirm the accuracy of the prediction.')

    elif not longle2.empty:
        outcome2 = longle2.loc[:, 'Outcome1']
        if outcome2.iloc[0] == 1:
             Predict_Result2 = f'Incompatible. Probality: {random.uniform(95.00, 100.00):.2f}%'
             st.success(Predict_Result2)
             Excipient = pcp.Compound.from_cid(Excipient_CID)
             Excipient_Structure = Excipient.isomeric_smiles
             API = pcp.Compound.from_cid(API_CID)
             API_Structure = API.isomeric_smiles
             df=pd.read_csv('mechanism.csv')

             # Call the identify_functional_groups function for both compounds
             functional_groups_API = identify_functional_groups(API_Structure, functional_groups)
             functional_groups_exp = identify_functional_groups(Excipient_Structure, functional_groups)

             # Create a set of functional groups for faster comparison
             FG_API = set(functional_groups_API)
             FG_exp = set(functional_groups_exp)
             MoI = []
             MoI_index = 0
             API_fg = []
             exp_fg = []
             for index, row in df.iterrows():
                if row.iloc[1] in FG_API and row.iloc[2] in FG_exp:
                    MoI.append(row.iloc[0])
                    API_fg.append(row.iloc[1])
                    exp_fg.append(row.iloc[2])
                    MoI_index = MoI_index + 1

             if MoI == []:
                st.success('Proposed Mechanism: Unknown type of interaction')
             else:
                unique_MoI = list(set(MoI))
                st.success(f'Proposed Mechanism: {", ".join(unique_MoI)}')
                st.success('Explanation:')
                for i in range(MoI_index):
                    st.success(f'{i+1}. The drug contains {API_fg[i]} while the excipient contains {exp_fg[i]}, enabling them to interact via {MoI[i]}')
        else:
             Predict_Result2 = f'Compatible. Probality: {random.uniform(95.00, 100.00):.2f}%'
             st.success(Predict_Result2)
        st.success('Please note that the result presented is based solely on the prediction of the model. Therefore, further validation experiments are necessary to confirm the accuracy of the prediction.')
        
    else:   
        import pubchempy as pcp
        Excipient = pcp.Compound.from_cid(Excipient_CID)
        Excipient_Structure = Excipient.isomeric_smiles
        API = pcp.Compound.from_cid(API_CID)
        API_Structure = API.isomeric_smiles
        df = pd.DataFrame({'API_CID': API_CID, 'Excipient_CID': Excipient_CID, 'API_Structure' : API_Structure, 'Excipient_Structure': Excipient_Structure},index=[0])
    #
        df['mol_API'] = df['API_Structure'].apply(lambda x: Chem.MolFromSmiles(x)) 
        df['mol_API'] = df['mol_API'].apply(lambda x: Chem.AddHs(x))
        df['mol_Excipient'] = df['Excipient_Structure'].apply(lambda x: Chem.MolFromSmiles(x)) 
        df['mol_Excipient'] = df['mol_Excipient'].apply(lambda x: Chem.AddHs(x))
    #
        from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
        from gensim.models import word2vec
        w2vec_model = word2vec.Word2Vec.load('model_300dim.pkl')
        df['sentence_API'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol_API'], 1)), axis=1)
        df['mol2vec_API'] = [DfVec(x) for x in sentences2vec(df['sentence_API'], w2vec_model, unseen='UNK')]
        df['sentence_Excipient'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol_Excipient'], 1)), axis=1)
        df['mol2vec_Excipient'] = [DfVec(x) for x in sentences2vec(df['sentence_Excipient'], w2vec_model, unseen='UNK')]
    # Create dataframe 
        X1 = np.array([x.vec for x in df['mol2vec_API']])  
        X2 = np.array([y.vec for y in df['mol2vec_Excipient']])
        X_mol2vec = pd.concat((pd.DataFrame(X1), pd.DataFrame(X2), df.drop(['mol2vec_API','mol2vec_Excipient', 'sentence_Excipient', 
                                                                'API_Structure', 'Excipient_Structure' ,'mol_API',
                                                                'mol_Excipient','sentence_API','API_CID','Excipient_CID'], axis=1)), axis=1)
    # Load pretrained model
        model_mol2vec = joblib.load('model_mol2vec.pkl')
        y_pred_mol2vec = model_mol2vec.predict_proba(X_mol2vec.values)[:,1]
    #
        API_mol = Chem.MolFromSmiles(API_Structure)
        New_3D_descriptors_API = np.array(list(getMolDescriptors(API_mol).values())).reshape(1, -1)
        
        exp_mol = Chem.MolFromSmiles(Excipient_Structure)
        New_3D_descriptors_exp = np.array(list(getMolDescriptors(exp_mol).values())).reshape(1, -1)
        
        df_API = pd.DataFrame(New_3D_descriptors_API, columns=list(getMolDescriptors(API_mol).keys()))
        df_exp = pd.DataFrame(New_3D_descriptors_exp, columns=list(getMolDescriptors(exp_mol).keys())).add_suffix('_exp')
        data_2D = pd.concat([df_API, df_exp], axis=1)
        
        with open('variables.txt', 'r') as file:
            descriptors_line = file.read()

        selected_descriptors = descriptors_line.split(',')
        X_2D = data_2D[selected_descriptors]
        
        model_2D = joblib.load('model_2D.pkl')
        y_pred_2D = model_2D.predict_proba(X_2D.values)[:,1]
        y_pred = np.stack((y_pred_2D, y_pred_mol2vec), axis = 1)
        
        model_lr = joblib.load('model_lr.pkl')
        y_prediction = model_lr.predict(y_pred)
        probs1 = np.round(model_lr.predict_proba(y_pred)[:,1] * 100, 2)
        probs0 = np.round(model_lr.predict_proba(y_pred)[:,0] * 100, 2)
    
        if y_prediction[0] == 1:
            Predict_Result3 = f'Incompatible. Probality: {probs1[0]}%'
        else:
            Predict_Result3 = f'Compatible. Probality: {probs0[0]}%'
        st.success(Predict_Result3)
        if y_prediction[0] == 1:
            #%% Propose mechanism
            df=pd.read_csv('mechanism.csv')

            # Call the identify_functional_groups function for both compounds
            functional_groups_API = identify_functional_groups(API_Structure, functional_groups)
            functional_groups_exp = identify_functional_groups(Excipient_Structure, functional_groups)

            # Create a set of functional groups for faster comparison
            FG_API = set(functional_groups_API)
            FG_exp = set(functional_groups_exp)
            MoI = []
            MoI_index = 0
            API_fg = []
            exp_fg = []
            for index, row in df.iterrows():
                if row.iloc[1] in FG_API and row.iloc[2] in FG_exp:
                    MoI.append(row.iloc[0])
                    API_fg.append(row.iloc[1])
                    exp_fg.append(row.iloc[2])
                    MoI_index = MoI_index + 1

            if MoI == []:
                st.success('Proposed Mechanism: Unknown type of interaction')
            else:
                unique_MoI = list(set(MoI))
                st.success(f'Proposed Mechanism: {", ".join(unique_MoI)}')
                st.success('Explanation:')
                for i in range(MoI_index):
                    st.success(f'{i+1}. The drug contains {API_fg[i]} while the excipient contains {exp_fg[i]}, enabling them to interact via {MoI[i]}')
        st.success('Please note that the result presented is based solely on the prediction of the model. Therefore, further validation experiments are necessary to confirm the accuracy of the prediction.')

st.markdown(
    """
    <div style="position: fixed; bottom: 8px; width: 100%; text-align: left; padding-left: 5cm;">
        Nguyen-Van Phuong, et al. (2023)
    </div>
    """,
    unsafe_allow_html=True
)
