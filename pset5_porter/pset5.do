clear all
import delimited "/Users/anyamarchenko/Documents/GitHub/IO_psets/pset5_porter/Porter.csv"

// Try a simple OLS regression of $\log(\text{QUANTITY})$ on a constant, $\log(\text{PRICE})$, \text{LAKES}, and (twelve of) the seasonal dummy variables?

g ln_q = ln(quantity)
g ln_p = ln(price)

reg ln_q ln_p i.lakes seas1-seas13, r

//Try doing the regression instead using instrumental variables with the COLLUSION variable as the instrument for PRICE. How does the reported price elasticity change? Is the estimate closer to that in Porter's paper or that in Ellison's paper and why? How do you interpret the coefficient on the LAKES variable? On the seasonal dummies? What is the $R^2$ of the regression and what do you make of it?

ivregress 2sls ln_q i.lakes seas1-seas13 (ln_p = collusion), vce(robust)

//Try the regression with the DM2 variable instead of COLLUSION as an instrument for price. In what way do the results look "worse" and why do you think this happens.

ivregress 2sls ln_q i.lakes seas1-seas13 (ln_p = dm2), vce(robust)


//Estimate a supply equation as in Porter and Ellison using the LAKES variable as an instrument for quantity. What does the magnitude of the coefficient on COLLUSION tell us about the effect of collusion on prices? What might the coefficient on QUANTITY in this regression indicate about the nature of costs in the JEC?

ivregress 2sls ln_p dm* collusion (ln_q = lakes), vce(robust)

//    Q_t = \alpha_0 + \alpha_1 P_t + \alpha_2 \text{Lakes}_t + \alpha_{3-14} \text{Seasxx}_{t} + \alpha_{15} \text{Lakes}_t P_t + U_{1_t}.
//Estimate the demand equation above and a supply equation motivated by the behavior of a monopolist with constant marginal costs using instrumental variables. (Try the Collusion variable and the Collusion variable interacted with the Lakes variable as the two instruments.)

// demand
ivregress 2sls quantity i.lakes seas1-seas13 (price price#lakes = collusion collusion#lakes), vce(robust)

// supply 
ivregress 2sls price dm* (quantity c.quantity#i.collusion = collusion collusion#lakes), vce(robust)
