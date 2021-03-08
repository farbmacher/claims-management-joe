# =============================================================================
# Replication files for 
# „An Explainable Attention Network for Fraud Detection in Claims Management“, 
# Helmut Farbmacher, Leander Löw, Martin Spindler,
# Journal of Econometrics
# =============================================================================

import os
import random
import argparse

from constants_imports import *
from generator_functions import *
from model_functions import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from warmup_scheduler import GradualWarmupScheduler

parser = argparse.ArgumentParser(description='Hyperopt')

parser.add_argument('-s', '--seed', default=133141, type=int,
                    help='random seed to start hyperopt')
parser.add_argument('-d', '--device',
                    help='the device to run on can be cuda:0 - how many present or cpu ')

def main():
    args = parser.parse_args()
    hyperopt(args.seed,args.device)

def hyperopt(seed=1235554,DEVICE="cuda:0",DATA_PATH=None):
 
     space = [Real(0, 0.5, "uniform", name='DROP_OUT'),
             Real(0.00001, 0.0001, "uniform", name='WEIGHT_DECAY'),
             Categorical([0.0001], name='LEARNING_RATE'),
             Categorical([64,128,256], name='MODEL_DIM'),
             Categorical(["no","selfa"], name='EXTRACT'),
             Integer(1, 5, name='NLAYERS_top'),
             Integer(1, 5, name='NLAYERS'),
             Categorical([32,64,128], name='BATCH_SIZE'),
             Categorical([1], name='HEADS'),
             Categorical(["attention"], name='AGGREGATION'),
             Categorical([False], name='LN_LOW'),
             Categorical([False], name='LN_HIGH'),
             ]
 
     @use_named_args(space)
     def hyper_fit(**params):
         value = training_run(**params)
         return -value
     res_gp = gp_minimize(hyper_fit, space, n_calls=100, random_state=seed,verbose=True)
     print(res_gp)

# this defines a single run given optimal hyperparameters
def training_run(
        WEIGHT_DECAY=0.00003,
        LEARNING_RATE=0.0001,
        MODEL_DIM=128,
        EXTRACT="selfa",
        NLAYERS=1,
        NLAYERS_top=4,
        BATCH_SIZE=128,
        DROP_OUT=0.164,
        COST_TP=10,
        HEADS=1,
        AGGREGATION="attention",
        LN_LOW=False,
        LN_HIGH=False,
        DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    configuration = {
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": round(WEIGHT_DECAY,5),
        "MODEL_DIM": MODEL_DIM,
        "EXTRACT": EXTRACT,
        "NLAYERS": NLAYERS,
        "NLAYERS_top": NLAYERS_top,
        "HEADS": HEADS,
        "AGGREGATION": AGGREGATION,
        "BATCH_SIZE": int(BATCH_SIZE),
        "DROP_OUT": round(DROP_OUT,3),
        "COST_TP": COST_TP,
        "LN_LOW": LN_LOW,
        "LN_HIGH": LN_HIGH,
    }
    
    WEIGHT_DECAY=round(WEIGHT_DECAY,5)
    DROP_OUT=round(DROP_OUT,3)
    BATCH_SIZE=int(BATCH_SIZE)
 
    print('Using device:', DEVICE)
    print()
  
    # TRAIN VAL TEST SPLIT
    data_table = pd.read_pickle(DATA_PATH + "data_formatted.pkl")
    train = data_table[data_table["train_val_test_split"] == 1]
    test = data_table[data_table["train_val_test_split"] == 3]
    val = data_table[data_table["train_val_test_split"] == 2]
    train = train.sort_index()
    test = test.sort_index()
    val = val.sort_index()

    # SET UP FOR TRACKNG WITH TENSORBOARD
    tf_folder_name="tensorboard_files_6//"
    try:
        os.mkdir(DATA_PATH + tf_folder_name)
    except:
        pass

    trial_name = str(random.randrange(1000000)) +"//"
    logd = DATA_PATH + tf_folder_name + trial_name
    print("--------------------------------------------------")
    print(logd)
    writer = SummaryWriter(
        log_dir=logd)

    # GETTING THE GENERATORS
    olg_train = one_level_generator(train)
    olg_gen_train = DataLoader(olg_train, shuffle=True, num_workers=CORES_TO_USE, collate_fn=my_collate,
                               batch_size=BATCH_SIZE,
                               drop_last=False)

    olg_test = one_level_generator(test)
    olg_gen_test = DataLoader(olg_test, shuffle=False, num_workers=CORES_TO_USE, collate_fn=my_collate,
                              batch_size=BATCH_SIZE,
                              drop_last=False)
    
    olg_val = one_level_generator(val)
    olg_gen_val = DataLoader(olg_val, shuffle=False, num_workers=CORES_TO_USE, collate_fn=my_collate,
                             batch_size=BATCH_SIZE,
                             drop_last=False)

    # SETTING UP MODEL
    fm = fraud_model(model_dim=MODEL_DIM
                     , feature_method=EXTRACT
                     , heads=HEADS
                     , drop=DROP_OUT
                     , aggregation=AGGREGATION
                     , n_layers_extract=NLAYERS
                     , n_layers_fc=NLAYERS_top
                     , ln_high=LN_HIGH, ln_low=LN_LOW).to(DEVICE)

    total_step = 0
    max_epochs = 100
    best_score = 0
    max_wait = 20
    wait = 0

    optimizer=torch.optim.Adam(fm.parameters(),
                                              lr=LEARNING_RATE,
                                              weight_decay=WEIGHT_DECAY)
    # THINGS USED FOR THE MODEL LOOP
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10, after_scheduler=scheduler_cosine)

    all_result_dict = {}
    for epoch in range(max_epochs):
        epoch_dict = {}

        total_step, model, prediction_list, output_list, label_list, scaling_list = train_eval(olg_gen_train, writer,
                                                                                               optimizer, total_step,
                                                                                               fm, COST_TP, True,DEVICE)
        # STORING RESULTS FROM TRAINING
        epoch_dict.update({
            "train_aupr": get_aucpr(label_list, prediction_list),
            "train_auroc": get_aucroc(label_list, prediction_list),
            "train_profit_score": get_profit_score(label_list, scaling_list, prediction_list)
        })

        # THE VALIDATION LOOP
        total_step, model, prediction_list, output_list, label_list, scaling_list = train_eval(olg_gen_val, writer,
                                                                                               optimizer, total_step,
                                                                                               fm, COST_TP, False,DEVICE)
        # STORING RESULTS FROM VALIDATION
        epoch_dict.update({
            "val_aupr": get_aucpr(label_list, prediction_list),
            "val_auroc": get_aucroc(label_list, prediction_list),
            "val_profit_score": get_profit_score(label_list, scaling_list, prediction_list),
        })

        for score_name, score_value in epoch_dict.items():
            writer.add_scalar(score_name, score_value, total_step)
        scheduler.step(epoch_dict["val_profit_score"])

        # EARLY STOPPING
        if epoch_dict["val_profit_score"] > best_score:
            best_score = epoch_dict["val_profit_score"]
            epoch_dict.update({"model": model.state_dict(), "configuration": configuration, "best_score": best_score,
                               "total_step": total_step, "wait": wait,"epoch":epoch,
                               "model_definition": trial_name,"test_predictions":get_predictions_with_ids(test,model,DEVICE)})
            torch.save(epoch_dict, DATA_PATH + tf_folder_name + trial_name + "model_file.pt")
            
            wait = 0
        else:
            wait = wait + 1
            if wait == max_wait:
                break

    return best_score

def train_eval(generator, sum_w, optimizers, total_step, model, cost_tp, train=True,DEVICE=None):
    prediction_list = []
    output_list = []
    label_list = []
    scaling_list = []
    loss = nn.BCELoss(reduction="none")
    gen = iter(generator)
    for label, scaling, numeric_batch, categorical_batch in tqdm(gen):
        label = label.to(DEVICE)
        scaling = scaling.to(DEVICE)
        numeric_batch = numeric_batch.to(DEVICE)

        for x, y in categorical_batch.items():
            categorical_batch[x] = categorical_batch[x].to(DEVICE)

        prediction = model(categorical_batch, numeric_batch).squeeze(1)
        label[(scaling > 0) * (scaling < cost_tp)] = 0
        scaling = torch.abs(scaling - cost_tp)
        output = loss(prediction, label.float())
        output = output * (scaling / 1000)
        output = output.mean()
        all_params = model.parameters()

        if train == True:
            output.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 10)
            optimizers.step()
            optimizers.zero_grad()
            total_step = total_step + 1

        output = output.detach().cpu().numpy()
        prediction = prediction.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        scaling = scaling.detach().cpu().numpy()

        if train == True:
            name_r = "train"
        else:
            name_r = "val"

        if train == True:
            sum_w.add_scalar("Loss/" + name_r, output, total_step)

        prediction_list.append(prediction)
        output_list.append(output)
        label_list.append(label)
        scaling_list.append(scaling)

    prediction_list = np.concatenate(prediction_list, axis=0)
    output_list = np.array(output_list)
    label_list = np.concatenate(label_list, axis=0)
    scaling_list = np.concatenate(scaling_list, axis=0)

    return total_step, model, prediction_list, output_list, label_list, scaling_list

def get_profit_score(label_list, scaling_list, prediction_list,thresh=None):
    prof_list = []
    thresh_list = []
    for thresh in np.arange(0.3, 0.6, 0.01):
        max_possible_profit = np.sum(label_list * scaling_list)
        binary_predictions = (prediction_list > thresh)
        true_positives = binary_predictions * label_list
        benefit_of_tp = np.sum(true_positives * scaling_list)
        false_positives = binary_predictions * (label_list == 0)
        cost_of_false_positives = np.sum(false_positives * scaling_list)
        score = (benefit_of_tp - cost_of_false_positives) / max_possible_profit
        prof_list.append(score)
        thresh_list.append(thresh)

    score = np.max(prof_list)
    best_thresh = thresh_list[np.argmax(prof_list)]

    print(
        f"Best thresh: {round(best_thresh,2)} Total score of {round(score,2)} with a benefit of tp of {round(benefit_of_tp,2)} a cost of fp of {round(cost_of_false_positives,2)} and a max profit of {round(max_possible_profit,2)}")
    return score

def get_aucpr(label, pred):
    if len(np.unique(label)) == 1:
        print("only one class present returning default score of 0")
        return 0
    precision, recall, thresh = precision_recall_curve(label, pred, pos_label=1)
    return auc(recall, precision)

def get_aucroc(label, pred):
    if len(np.unique(label)) == 1:
        print("only one class present returning default score of 0.5")
        return 0.5
    return roc_auc_score(label, pred)

def get_predictions_with_ids(data_table,model,DEVICE):
# GETTING THE GENERATORS
    olg_val = one_level_generator(data_table)
    olg_gen_val = DataLoader(olg_val, shuffle=False, num_workers=CORES_TO_USE, collate_fn=my_collate,
                             batch_size=128,
                             drop_last=False)
    prediction_list = []
    output_list = []
    label_list = []
    scaling_list = []
    cost_tp=10
    gen = iter(olg_gen_val)
    for label, scaling, numeric_batch, categorical_batch in tqdm(gen):
        label = label.to(DEVICE)
        scaling = scaling.to(DEVICE)
        numeric_batch = numeric_batch.to(DEVICE)
        for x, y in categorical_batch.items():
            categorical_batch[x] = categorical_batch[x].to(DEVICE)
        prediction = model(categorical_batch, numeric_batch).squeeze(1)
        label[(scaling > 0) * (scaling < cost_tp)] = 0
        scaling = torch.abs(scaling - cost_tp)
        prediction = prediction.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        scaling = scaling.detach().cpu().numpy()
        prediction_list.append(prediction)
        label_list.append(label)
        scaling_list.append(scaling)

    prediction_list = np.concatenate(prediction_list, axis=0)
    output_list = np.array(output_list)
    label_list = np.concatenate(label_list, axis=0)
    scaling_list = np.concatenate(scaling_list, axis=0)
    pred_with_id=pd.Series(prediction_list,index=data_table.index.unique())   
    return pred_with_id

def gen_features(data_table, model,DEVICE,batch_size=128):
    olg_val = one_level_generator(data_table)
    olg_gen_val = DataLoader(olg_val, shuffle=False, num_workers=CORES_TO_USE, collate_fn=my_collate,
                             batch_size=batch_size,
                             drop_last=False)
    prediction_list = []
    gen = iter(olg_gen_val)
    for label, scaling, numeric_batch, categorical_batch in tqdm(gen):
        numeric_batch = numeric_batch.to(DEVICE)
        for x, y in categorical_batch.items():
            categorical_batch[x] = categorical_batch[x].to(DEVICE)
        _,prediction = model(categorical_batch, numeric_batch,return_hidden=True)
        prediction = prediction.detach().cpu().numpy()
        prediction_list.append(prediction)
    prediction_list = np.concatenate(prediction_list, axis=0)
    pred_with_id = pd.DataFrame(prediction_list, index=data_table.index.unique())
    return pred_with_id

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        self.lr=lr
            
if __name__ == "__main__":
    main()              # performs hyperparameter search
#    training_run()     # performs only one run with optimal hyperparameters
