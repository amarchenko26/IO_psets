* Pset 2 IO 
* Sept 2024

* Create the dataset with variables: Price, Q_in_Calif, Q_in_Hawaii
clear
input price q_calif q_hawaii
10 130 31
11 106 27
12 105 31
14 100 24
15 60 24
16 70 25
17 65 18
18 60 23
20 48 21
22 28 14
24 12 18
25 2 14
26 1 10
30 0 9
end

* Label the variables
label var price "Price"
label var q_calif "Quantity in California"
label var q_hawaii "Quantity in Hawaii"

* Display the dataset
list

******* CALI
reg price q_calif, r
local b = round(_b[q_calif], .001)  // slope
local a = round(_b[_cons], .001)       // intercept

* Create a note with the regression equation
local equation "Demand (solid line): price = `b' Q + `a'"
local mr "Marginal revenue (dashed line): MR = -0.274 Q + `a'"

* Generate the linear fit plot
twoway (scatter price q_calif) (lfit price q_calif) ///
		(function y = -0.274 * x + 26.251, range(0 90) lpattern(dash)), ///
        title("Estimate of linear demand curve in California") ///
        xlabel(10(10)130) ylabel(0(10)30) ytitle("Price") ///
        text(5 40 "`equation'", size(small) color(black)) ///
        text(2 40 "`mr'", size(small) color(black)) ///
		yline(10) ///
		yline(18.1, lcolor(blue)) ///
        legend(off)
		
graph export "/Users/anyamarchenko/Desktop/pset2_io/demand_ca.png", replace

		
******* HAWAII
reg price q_hawaii, r
local b: display %6.3f _b[q_hawaii]  // formatted slope
local a = round(_b[_cons], .001)       // intercept

* Create a note with the regression equation
local equation "Demand (solid line): price = `b' Q + `a'"
local mr "Marginal revenue (dashed line): MR = -1.630 Q + `a'"

* Generate the linear fit plot
twoway (scatter price q_hawaii) (lfit price q_hawaii) ///
		(function y = -1.630 * x + 35.388, range(8 17) lpattern(dash)), ///
        title("Estimate of linear demand curve in Hawaii") ///
        xlabel(5(5)35) ylabel(0(10)30) ytitle("Price") ///
        text(5 20 "`equation'", size(small) color(black)) ///
		text(2 20 "`mr'", size(small) color(black)) ///
		yline(10) ///
		yline(22.69, lcolor(blue)) ///
        legend(off)
		
graph export "/Users/anyamarchenko/Desktop/pset2_io/demand_hi.png", replace

		
		
		
		
		