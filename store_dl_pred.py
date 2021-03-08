# =============================================================================
# Replication files for 
# „An Explainable Attention Network for Fraud Detection in Claims Management“, 
# Helmut Farbmacher, Leander Löw, Martin Spindler,
# Journal of Econometrics
# =============================================================================

from constants_imports import *

tf_folder_name="tensorboard_files_6/"
all_folders = os.listdir(DATA_PATH + tf_folder_name)

if '.DS_Store' in all_folders:
    all_folders.remove('.DS_Store')

bestscore = 0
for fil in all_folders:
    model = torch.load(DATA_PATH + tf_folder_name + fil + "/model_file.pt")
    score = model["val_profit_score"]
    if score>bestscore:
        bestfil = fil
        bestscore = score
        bestmodel = model
        bestconfig = model["configuration"]

print(bestfil)
print(bestconfig)   
bestmodel["test_predictions"].to_csv('test_pred_dl.csv', mode='a', index=True)


