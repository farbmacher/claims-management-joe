# =============================================================================
# Replication files for 
# „An Explainable Attention Network for Fraud Detection in Claims Management“, 
# Helmut Farbmacher, Leander Löw, Martin Spindler,
# Journal of Econometrics
# =============================================================================

from constants_imports import *
import warnings

info_dict=load_pickle(DATA_PATH+"info_dict.pkl")

class one_level_generator(Dataset):
    def __init__(self,data_table,return_id=False):
        self.numerics=info_dict["numerics"]
        self.categoricals=info_dict["categoricals"]
        self.LABEL_COL=info_dict["LABEL_COL"]
        data_table=data_table[self.numerics+self.categoricals+[self.LABEL_COL]]

        for x in self.categoricals:
            catcodes = data_table[x].cat.codes
            del data_table[x]
            data_table[x] = catcodes + 2

        self.data_table=data_table
        self.original_id_list=list(data_table.index.unique())
        self.max_numerics=info_dict["max_numerics"]
        self.return_id=return_id
    
    def __getitem__(self, index,max_past=500):
        original_id=self.original_id_list[index]
        single_case=self.data_table.loc[[original_id]]
        if single_case.shape[0]>max_past:
            single_case=single_case.sample(max_past)
        single_case_numeric=single_case[self.numerics]
        single_case_categorical=single_case[self.categoricals]
        single_case_label=single_case[self.LABEL_COL].iloc[0]        
        single_case_numeric=single_case_numeric/self.max_numerics
        single_case_numeric=single_case_numeric.values
        return single_case_label,single_case_numeric,single_case_categorical.values

    def __len__(self):
        return len(self.original_id_list)

def my_collate(batch):
    labels = np.stack([x[0]for x in batch])
    bsize = len(labels)
    first_level_numerics = [x[1]for x in batch]
    first_level_categoricals = [x[2] for x in batch]
    num_numerics=first_level_numerics[0].shape[1]
    num_categoricals=first_level_categoricals[0].shape[1]
    variable_shape= max([x.shape[0] for x in first_level_categoricals])
    padded_numerics = np.zeros(shape=(bsize, variable_shape , num_numerics))
    padded_categorical = np.zeros(shape=(bsize, variable_shape , num_categoricals))
    for i, item_1 in enumerate(first_level_numerics):
        n_games=item_1.shape[0]
        if len(item_1) > 0:
            padded_numerics[i,-n_games:, :] = item_1
    for i, item_1 in enumerate(first_level_categoricals):
        n_games=item_1.shape[0]
        if len(item_1) > 0:
            padded_categorical[i,-n_games:, :] = item_1

    padded_categorical={x:torch.tensor(y).long() for x,y in zip(info_dict['categoricals'],np.split(padded_categorical,num_categoricals,axis=2))}
    padded_numerics=torch.tensor(padded_numerics).float()
    scaling=torch.tensor(labels)
    labels=torch.tensor(labels>0).long()
    return labels ,scaling ,padded_numerics ,padded_categorical

if __name__=="__main__":
    info_dict = load_pickle(DATA_PATH + "info_dict.pkl")
    data_table = pd.read_pickle(DATA_PATH + "data_formatted.pkl")

    data_table = data_table.set_index(info_dict['ID_COL']).sort_index()
    olg = one_level_generator(data_table)
    it=iter(olg)
    test_batch = next(it)
    olg_gen = DataLoader(olg, shuffle=True, num_workers=6, collate_fn=my_collate, batch_size=128, drop_last=True)
    for ge in tqdm(olg_gen):
        pass
    