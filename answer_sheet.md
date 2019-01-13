# Answer Sheet
The number below correspond to the questions provided [here](https://nipunbatra.github.io/teaching/ml-spring-19/hw/1.pdf) as an assignment to my Machine Learning course at IIT Gandhinagar.

### Answers 1
* Assuming that we have a decision tree algorithm that splits a categorical feature's node such that the number of children created are equal to the possible categories of that feature. If this is the case, then the decision tree would definietly have the first question on the `Day #`. This would create a total of 14 bins and each bin would be containing one sample.
This would result in a "Perfect" Decision tree that is able to classify with no error on the training data, only in 1 depth.
But actually this is method of including `Day #` in training is foolish as there is no correlation with the Day number and whether a person plays tennis or not. This would perform really badly on test data as the decision tree learnt sothing that has `0` correlation with the decision of playing tennis or not.

	Therefore I feel choosing to include `Day #` in the training set would not be helpful at all and would actually be bad for us.

* [Ipython notebook]()
One of the basic way to remove `NaN` is to replace the `NaN` with `mean`, `mode` or `median`. Since outlook is a categorical feature `mode` or `median` would make more sense.
We replace ``Day-3``'s outlook with the `mode` = `rainy` or `sunny` as both of them occur 5 times.

	We now use our implementation