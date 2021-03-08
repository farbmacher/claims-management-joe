////////////////////////////////////////////////////////////////////////////
// An Explainable Attention Network for Fraud Detection in Claims Management
// Helmut Farbmacher, Leander Löw and Martin Spindler
// Journal of Econometrics
////////////////////////////////////////////////////////////////////////////

clear all
set more off

cd "/Users/helmut/Desktop/Code JoE/"

import delimited "test_pred_dl.csv", encoding(ISO-8859-1)
rename v1 id
rename v2 pr_dl
save "test_pred_dl.dta", replace

clear
import delimited "test_pred_gbm.csv", encoding(ISO-8859-1)
rename v1 id
rename v2 pr_gbm
save "test_pred_gbm.dta", replace



