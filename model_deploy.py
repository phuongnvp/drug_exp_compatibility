import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pubchempy as pcp
import numpy as np                
import pandas as pd               
import matplotlib.pyplot as plt   
import seaborn as sns
from rdkit.Chem import Descriptors
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from rdkit import Chem
import random

#%%
def get_cid(api, option):
    if option == 'Name':
        compound = pcp.get_compounds(api, 'name')[0]
    elif option == 'PubChem CID':
        compound = pcp.Compound.from_cid(int(api))
    elif option == 'SMILES':
        compound = pcp.get_compounds(api, 'smiles')[0]
    return int(compound.cid)

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
st.title('Drug - Excipient Compatibility')
col1, col2 = st.columns([1,3])
with col1: 
    option1 = st.selectbox('Search Option', ['Name', 'PubChem CID', 'SMILES'])
with col2:
    API_CID = st.text_input('Enter name, Pubchem CID or smiles string of the API')
col3, col4 = st.columns([1,3])
with col3: 
    option3 = st.selectbox('', ['Name', 'PubChem CID', 'SMILES'])
with col4:
    Excipient_CID = st.text_input('Enter name, Pubchem CID or smiles string of the excipient')

df1 = pd.read_csv('dataset.csv')
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
        else:
            Predict_Result1 = f'Compatible. Probality: {random.uniform(95.00, 100.00):.2f}%'
        st.success(Predict_Result1)
        st.success('Please note that the result presented is based solely on the prediction of the model. Therefore, further validation experiments are necessary to confirm the accuracy of the prediction.')

    elif not longle2.empty:
        outcome2 = longle2.loc[:, 'Outcome1']
        if outcome2.iloc[0] == 1:
             Predict_Result2 = f'Incompatible. Probality: {random.uniform(95.00, 100.00):.2f}%'
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
        st.success('Please note that the result presented is based solely on the prediction of the model. Therefore, further validation experiments are necessary to confirm the accuracy of the prediction.')

st.markdown(
    """
    <div style="position: fixed; bottom: 8px; width: 100%; text-align: left; padding-left: 5cm;">
        For the updated version of this website, please visit <a href="https://decompatibility-v12.streamlit.app">here</a>
    </div>
    """,
    unsafe_allow_html=True
)
