////////////////////////////////////////////////////////////////////////////
// An Explainable Attention Network for Fraud Detection in Claims Management
// Helmut Farbmacher, Leander Löw and Martin Spindler
// Journal of Econometrics
////////////////////////////////////////////////////////////////////////////

clear all
set more off

cd "/Users/helmut/Desktop/Code JoE/"

//parameters to choose:
local costs=10				//manual auditing costs per claim
local auditing_max=500		//maximal number of claims considered for potential auditing

////////////

use data_public.dta

bysort id: egen nuitems=max(item)

collapse (mean) korrektur fraud betrag nuitems, by(id)

merge 1:1 id using test_pred_dl.dta		//prediction from dl model
assert (_merge==2)==0
keep if _merge==3			//keep only test data
drop _merge
merge 1:1 id using test_pred_gbm.dta	//prediction from boosted trees
assert (_merge==3)
drop _merge
compress

save pi_temp.dta, replace

//randomly drawing from the subset of fraudulent claims: expected gains (oracle 1)
use pi_temp.dta
sum korrektur
scalar rsum=r(sum)
sum fraud if fraud==1
gen korrektur_rand_fraud=rsum/r(N)
sum korrektur_rand_fraud
gen index_oracle=_n

gen costs = `costs'
gen gain=korrektur_rand_fraud-costs
gen pi_rand_oracle=korrektur_rand_fraud[1]-costs[1] in 1
replace pi_rand_oracle=pi_rand_oracle[_n-1]+gain if _n>1

keep if _n<=`auditing_max'
rename index_oracle s
keep s pi_rand_oracle
save pi_rand_oracle_temp.dta, replace

//draw biggest frauds (oracle 2)
use pi_temp.dta
gsort -korrektur
gen index_oracle=_n

gen costs = `costs'
gen gain=korrektur-costs
gen pi_oracle=korrektur[1]-costs[1] in 1
replace pi_oracle=pi_oracle[_n-1]+gain if _n>1

keep if _n<=`auditing_max'
rename index_oracle s
keep s pi_oracle
merge 1:1 s using pi_rand_oracle_temp.dta
assert (_merge==3)
drop _merge
save pi_oracle_temp.dta, replace

//using boosted trees
use pi_temp.dta
gsort -pr_gbm
gen index_gbm=_n

gen costs = `costs'
gen gain=korrektur-costs
gen pi_gbm=korrektur[1]-costs[1] in 1
replace pi_gbm=pi_gbm[_n-1]+gain if _n>1

keep if _n<=`auditing_max'
rename index_gbm s
keep s pi_gbm
merge 1:1 s using pi_oracle_temp.dta
assert (_merge==3)
drop _merge
save pi_gbm_temp.dta, replace

//using deep learning
use pi_temp.dta
gsort -pr_dl
gen index_dl=_n

gen costs = `costs'
gen gain=korrektur-costs
gen pi_dl=korrektur[1]-costs[1] in 1
replace pi_dl=pi_dl[_n-1]+gain if _n>1

keep if _n<=`auditing_max'
rename index_dl s
keep s pi_dl
merge 1:1 s using pi_gbm_temp.dta
assert (_merge==3)
drop _merge

twoway (line pi_oracle s)(line pi_rand_oracle s if s<=485) (line pi_gbm pi_dl s, ytitle("{&pi}", orientation(horizontal)) ///
	ylabel("0") yscale(range(-10000 60000)) ///
	xtitle("N{subscript:{&tau}}") legend(order(1 "Oracle (C&P)" 2 "Oracle (C)" 3 "Boosted Trees" 4 "Deep Learning") row(2)) yline(0))

