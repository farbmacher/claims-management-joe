////////////////////////////////////////////////////////////////////////////
// An Explainable Attention Network for Fraud Detection in Claims Management
// Helmut Farbmacher, Leander Löw and Martin Spindler
// Journal of Econometrics
////////////////////////////////////////////////////////////////////////////

clear all
set more off

cd "/Users/helmut/Desktop/Code JoE/"

local maxN=500
local top=49
			
use data_public.dta

bysort id: egen nuitems=max(item)

collapse (mean) korrektur fraud betrag nuitems, by(id)

merge 1:1 id using test_pred_dl.dta		//prediction from dl model
assert (_merge==2)==0
keep if _merge==3		//keep only test data
drop _merge
merge 1:1 id using test_pred_gbm.dta	//prediction from boosted trees
assert (_merge==3)
drop _merge
compress

gsort -korrektur
gen index_oracle=_n

gsort -pr_gbm
gen index_gbm=_n

gsort -pr_dl
gen index_dl=_n

sum fraud if fraud==1
local nu_frauds=r(N)

sort index_oracle
keep if _n<=`maxN'

gen top1_gbm=.
gen top1_dl=.
scalar temp=0
forvalues i=1(1)`maxN' {
	scalar temp=scalar(temp)+(index_gbm[`i']<=`top')
	qui replace top1_gbm=scalar(temp)/`top' in `i'
}
scalar temp=0
forvalues i=1(1)`maxN' {	
	scalar temp=scalar(temp)+(index_dl[`i']<=`top')
	qui replace top1_dl=scalar(temp)/`top' in `i'
}

twoway (line top1_gbm top1_dl index_oracle, xtitle("N{subscript:{&tau}}") ytitle("") scheme(s2mono) ///
	legend(order(1 "Boosted Trees" 2 "Deep Learning") row(1)) yline(1) ylabel(0(0.5)1, angle(h)))

