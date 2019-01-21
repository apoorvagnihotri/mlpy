# Answer Sheet
The number below correspond to the questions provided [here](https://nipunbatra.github.io/teaching/ml-spring-19/hw/2.pdf) as an assignment to my Machine Learning course at IIT Gandhinagar.

[**Jupyter Notebook**](https://github.com/k0pch4/decision-trees/blob/master/usage/decision-tree/assignment1.ipynb)

### Answer 1

* Assuming that we have a decision tree algorithm that splits a categorical feature's node such that the number of children created is equal to the possible categories of that feature. If this is the case, then the decision tree would definitely have the first question on the `Day #`. This would create a total of 14 bins and each bin would be containing one sample.
This would result in a "Perfect" Decision tree that is able to classify with no error on the training data, only in 1 depth.
But actually, to include `Day #` in features that influence classification is foolish. This is due to the reason that there is no correlation with ``Day #`` and whether a person plays tennis or not. This would perform really badly on test data as the decision tree learned something that has `0` correlation with the decision of playing tennis or not.

    Therefore I feel choosing to include `Day #` in the training set would not be helpful at all and would actually be bad for us.

* One of the basic ways to remove `NaN` is to replace the `NaN` with `mean`, `mode` or `median`. Since outlook is a categorical feature `mode` or `median` would make more sense.
We replace ``Day-3``'s outlook with the `mode` = `rainy` or `sunny` as both of them occur 5 times.

    We now use our implementation to build a decision tree for the un-wrangled data and the original data. We see that decision tree learned is different both cases.

    ![](https://i.imgur.com/Xw8luj9.png)

---

### Answer 2

* [`src`](https://github.com/k0pch4/decision-trees/tree/master/src)
My Implementation imports a few files with custom functions and classes. Have a look at the `src` folder of the repository.

* Shows the usage of the implementation on IRIS dataset. The accuracy with this implementation of decision tree comes out to be `0.977` and it comes out to be `1.0` with the sklearn's implementation.

* The optimal ``depth`` in my implementation came out to be equal to `5` for the IRIS dataset. Have a look at the notebook above.

---

### Answer 3

Visit the above link to look at the demo of working on a regression problem using my implementation.

---

### Answer 4

Visit the above link to look at the difference in the performance with my implementation vs sklearn's.

---

### Answer 5

Visit the above link to look at the visualizations for the DT in my implementation.
I didn't use dtreeviz as the library is implemented such that it takes in SKLearn Objects, and I had implemented a graphical way to store the decision tree.

---

### Answer 6

---

### Answer 7
According the documentations of sklearn, we know that the complexity of Decision trees is devided in the following way:
`m` = # of Features
`n` = # of Samples
* Training - `m*n*log(n)`
* Predicting - `log(n)`

Since we are working with **small `m` and `n`** due to computational limits we are not able to truly appreciate the time complexity.

---
