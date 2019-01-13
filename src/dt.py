# '''
# Author: Apoorv Agnihotri

# This implementation would be slow on larger datasets.
# I would file the issues that make this implementation slow.

# Coding style has been inspired from the below link,
# https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
# Credits ^^^
# '''

# class DecisionNode:
#     '''Will record the Question to be asked, Parent (or not)
#     and the left and right children (DecisionNode/PredictionNode)
#     left_child contains true rows
#     right_child contains false rows
#     '''
#     def __init__(self, question, nodeL, nodeR):
#         self.question = question
#         self.left_child = nodeL
#         self.right_child = nodeR

# class PredictionNode:
#     '''This will contain label that is most common.
#     @add support for regression.
#     '''
#     def __init__(self, rows, RorC):
#         self.RorC = RorC
#         if RorC == 'c':
#             best_label = most_probable_label(rows)
#             self.label = best_label
#         else:
#             pass

#     def predict(self, row):
#         if self.RorC == 'c':
#             return self.label
#         else:
#             pass
#             # return self.average_val of the rows 

#     def average(self, rows):
#         # summ 
#         # self.average_val = summ
#         pass

# def classify(row, node):
#     if isinstance(node, PredictionNode): # tree end
#         return node.predict(row)
#     if node.question.is_satified(row):
#         return classify(row, node.left_child)
#     else:
#         return classify(row, node.right_child)