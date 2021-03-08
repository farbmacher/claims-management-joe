# =============================================================================
# Replication files for 
# „An Explainable Attention Network for Fraud Detection in Claims Management“, 
# Helmut Farbmacher, Leander Löw, Martin Spindler,
# Journal of Econometrics
# =============================================================================

import os
from constants_imports import *

def format_by_hand(DATA_PATH_ORIG=DATA_PATH_ORIG):
    all_files = os.listdir(DATA_PATH_ORIG)
    file_frame = []
    for fil in all_files:
        form_df = pd.read_csv(DATA_PATH_ORIG + fil, sep=";")
        file_frame.append(form_df)
    file_frame = pd.concat(file_frame)
    file_frame.head()

    # numerics are screwed in the sense that they have a , and are actually strings
    def format_numerics(file_frame, numeric_list=[]):
        for x in tqdm(numeric_list):
            file_frame[x] = file_frame[x].str.replace(",", ".")
            file_frame[x] = pd.to_numeric(file_frame[x])
        return file_frame

    NUMERIC_LIST = ["BETRAG"]
    file_frame = format_numerics(file_frame, NUMERIC_LIST + ["KORREKTUR", "FAKTOR"])
    file_frame["BETRAG"]=np.log(file_frame["BETRAG"]+1)
    file_frame["BETRAG"]=file_frame["BETRAG"].fillna(0)
    file_frame.FAKTOR = pd.cut(file_frame.FAKTOR, [1,1.15,1.8, 2.3,2.5, 9999.9], include_lowest=False, right=True)
    CATEGORICAL_LIST = ["FAKTOR", "NUMMER"]
    for x in tqdm(CATEGORICAL_LIST):
        file_frame[x] = file_frame[x].astype("category")
        file_frame = drop_low_category(file_frame, x)
        file_frame.rename(columns={x: x + "_category"}, inplace=True)
    for x in NUMERIC_LIST:
        file_frame.rename(columns={x: x + "_numeric"}, inplace=True)
    corrects = file_frame.groupby("ID").agg({"KORREKTUR": "max"})
    label_m = (corrects > 0).mean()
    label_s = corrects.sum()
    print(f"a total of {len(corrects)} cases with a total pay of {label_s.values} and a fraud rate of {label_m.values}")
    file_frame=file_frame[["ID","KORREKTUR","BETRAG_numeric","NUMMER_category","FAKTOR_category"]]
    file_frame=file_frame.set_index("ID")

    return file_frame

def get_info_dict(file_frame, LABEL_COL):
    allowed_dtypes = ['float64', 'float32', 'category']
    check_if_allowed_dtype = [x.name for x in file_frame.dtypes.drop([ LABEL_COL]) if
                              x.name not in allowed_dtypes]
    if len(check_if_allowed_dtype) > 0:
        print(
            f"found dtypes {check_if_allowed_dtype} in columns only allowed dtypes are {allowed_dtypes} please recode to either of those")
        assert False

    def get_vars_by_dtype(file_frame, TYPE_LIST):
        return [y for (y, x) in
                zip(file_frame.columns.drop([ LABEL_COL]), file_frame.dtypes.drop( LABEL_COL)) if
                x.name in TYPE_LIST]

    numeric_cols = get_vars_by_dtype(file_frame, ['float64', 'float32'])
    categorical_cols = get_vars_by_dtype(file_frame, ["category"])

    info_dict = {}
    info_dict["number_numerics"] = len(numeric_cols)
    info_dict["max_numerics"] = file_frame[numeric_cols].max(axis=0)
    info_dict["LABEL_COL"] = LABEL_COL
    info_dict["numerics"] = numeric_cols
    info_dict["categoricals"] = categorical_cols
    categorical_level_dict = {}
    for x in categorical_cols:
        categorical_level_dict.update(
            {x: len(file_frame[x].cat.categories) + 2})
    info_dict["categorical_level_dict"] = categorical_level_dict
    return info_dict

if __name__=="__main__":
      
    file_frame = format_by_hand()
    info_dict = get_info_dict(file_frame, "KORREKTUR")

    train_percent = .6
    validate_percent = .2
    np.random.seed(123)
    perm = np.random.permutation(file_frame.index.unique())
    m = len(file_frame.index.unique())
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    file_frame["train_val_test_split"] = 0
    file_frame.loc[perm[:train_end], "train_val_test_split"] = 1
    file_frame.loc[perm[train_end:validate_end], "train_val_test_split"] = 2
    file_frame.loc[perm[validate_end:], "train_val_test_split"] = 3

    print(file_frame["train_val_test_split"].value_counts())

    #stores info dict
    save_as_pickle(DATA_PATH + "info_dict.pkl", info_dict)
    file_frame.to_pickle(DATA_PATH + "data_formatted.pkl")
    