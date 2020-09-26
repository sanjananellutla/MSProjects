Description of the dataset:

Used the UCI car repository dataset as attached in the repository (http://archive.ics.uci.edu/ml/datasets/Car+Evaluation)

The Car Evaluation Database contains examples with the structural information removed, i.e., directly relates CAR to the six input attributes: buying, maint, doors, persons, lug_boot, safety.

Attribute Information:

Class Values:

	unacc, acc, good, vgood

Attributes:

	buying: vhigh, high, med, low.
	maint: vhigh, high, med, low.
	doors: 2, 3, 4, 5more.
	persons: 2, 4, more.
	lug_boot: small, med, big.
	safety: low, med, high

Description of code.py:

Applied Decision Tree, Bagging and Random Forest with 50 trees and with a maximum depth of five for each tree and mentioned the criteria used.

Showed how the accuracy changes along with the increase of number of trees in a plot.
	
Showed the variable importance in a plot in terms of decrease of used criteria and relative to 	the maximum.