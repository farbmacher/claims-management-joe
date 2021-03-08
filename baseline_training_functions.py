# =============================================================================
# Replication files for 
# „An Explainable Attention Network for Fraud Detection in Claims Management“, 
# Helmut Farbmacher, Leander Löw, Martin Spindler,
# Journal of Econometrics
# =============================================================================

import xgboost as xgb

from constants_imports import *
from generator_functions import *
from training_functions import *

def fit_baseline(depth):
    info_dict = load_pickle(DATA_PATH + "info_dict.pkl")
    data_table = pd.read_pickle(DATA_PATH + "data_formatted.pkl")
    train = data_table[data_table["train_val_test_split"] == 1]
    test = data_table[data_table["train_val_test_split"] == 3]
    val = data_table[data_table["train_val_test_split"] == 2]
    train = train.sort_index()
    test = test.sort_index()
    val = val.sort_index()

    id_name=data_table.index.get_level_values(0).name
    ag_train = aggregate(train, id_name, info_dict["LABEL_COL"])
    ag_val = aggregate(val, id_name, info_dict["LABEL_COL"])
    ag_test = aggregate(test, id_name, info_dict["LABEL_COL"])
    del train
    del val
    del test

    y_train, scaling_train = get_label_scaling(ag_train[info_dict["LABEL_COL"]].values)
    y_val, scaling_val = get_label_scaling(ag_val[info_dict["LABEL_COL"]].values)
    y_test, scaling_test = get_label_scaling(ag_test[info_dict["LABEL_COL"]].values)

    nums, cats = get_numerics(ag_train), get_categoricals(ag_train)
    col_names = list(ag_train[nums + cats].columns)

    # we cannot have this in the col_names (a restriction from xgboost)
    col_names = [x.replace(']', "") for x in col_names]

    ag_train = ag_train[nums + cats]
    ag_val = ag_val[nums + cats]
    ag_test = ag_test[nums + cats]
    data_dict = {
        "x_train": ag_train,
        "x_val": ag_val,
        "x_test": ag_test,
        "scaling_train": scaling_train,
        "scaling_test": scaling_test,
        "scaling_val": scaling_val,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "col_names": col_names
    }
    result_dict = fit_predict_xgboost(data_dict, depth)
    return result_dict

def get_label_scaling(scaling,cost_tp=10):
    label=scaling>0
    label[(scaling > 0) * (scaling < cost_tp)] = 0
    scaling = np.abs(scaling - cost_tp)
    return label, scaling

def simple_predict_xgboost(data_frame, gb):
    x_test = xgb.DMatrix(data_frame[gb.feature_names])
    preds = gb.predict(x_test)
    return preds

def aggregate(frame, index="erei#", label_col="Einsparungen"):
    frame = pd.get_dummies(frame, columns=get_categoricals(frame))
    cat_vars = get_categoricals(frame)
    num_vars = get_numerics(frame)

    print("aggregation")
    cat_dict_for_ag = dict(zip(cat_vars, np.repeat("sum", len(cat_vars))))
    ag_list_num = ["min", "max", "mean", "std"]
    num_dict_for_ag = dict(zip(num_vars, [ag_list_num for x in range(len(num_vars))]))
    num_dict_for_ag.update(cat_dict_for_ag)
    ag_dict_all = num_dict_for_ag
    ag_dict_all.update({
        label_col: "max", "train_val_test_split": "max"
    })
    frame = frame.groupby(index).agg(ag_dict_all)
    if isinstance(frame.columns, pd.core.indexes.multi.MultiIndex):
        frame.columns = ["_".join(col) for col in frame.columns]
    frame = frame.rename(columns={label_col + "_max": label_col
        , "train_val_test_split_max": "train_val_test_split"})
    return frame

def fit_predict_xgboost(input_dict, depth,return_pred=True):
    '''input_dict needs to be a dict with training data, validation data scalings.
    '''

    d_train = xgb.DMatrix(input_dict["x_train"], label=input_dict["y_train"]
                          , weight=input_dict["scaling_train"], feature_names=input_dict['col_names'])
    evals = xgb.DMatrix(input_dict["x_val"], label=input_dict["y_val"]
                        , weight=input_dict["scaling_val"], feature_names=input_dict['col_names'])
    x_test = xgb.DMatrix(input_dict["x_test"], feature_names=input_dict['col_names'])

    def profit_metric(pred, dtrain):
        profit = get_profit_score(dtrain.get_label(), dtrain.get_weight(), pred)
        return ("profit_metric", float(-profit))

    param = {"max_depth": depth, "eta": 0.1,
             "objective": "binary:logistic", "colsample_bylevel": np.log(2) / np.log(depth),
             "nthread": 3, "tree_method": "hist", "lambda": np.log(depth), "eval_metric": "auc"}

    gb = xgb.train(param, d_train, num_boost_round=1000, evals=[(evals, "valid")], early_stopping_rounds=50,
                   feval=profit_metric)
    d_train = xgb.DMatrix(input_dict["x_train"], label=input_dict["y_train"]
                          , weight=input_dict["scaling_train"], feature_names=input_dict['col_names'])
    preds_t = gb.predict(d_train)
    preds_v = gb.predict(evals)
    preds_test = gb.predict(x_test)

    preds_t=pd.Series(preds_t,index=input_dict["x_train"].index)
    preds_v = pd.Series(preds_v, index=input_dict["x_val"].index)
    preds_test = pd.Series(preds_test, index=input_dict["x_test"].index)

    importance_series = pd.Series(gb.get_score(importance_type="gain"))
    importance_series.sort_values(ascending=False, inplace=True)
    importance_series = importance_series / importance_series.sum()

    if return_pred==True:
        output_dict = {
            "trained_model": gb,
            "preds_t": preds_t,
            "preds_v": preds_v,
            "preds_test": preds_test,
            "importance_scores": importance_series
        }
    else:
        output_dict = {
            "trained_model": gb,
            "importance_scores": importance_series
        }

    del input_dict
    return output_dict

if __name__ == "__main__":

    out=fit_baseline(4)
    save_as_pickle( DATA_PATH+"baseline_file.py",out)
        
    # Stores predicted probs:
    out["preds_test"].to_csv('test_pred_gbm.csv', mode='a', index=True)
    