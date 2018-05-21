import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef

# given a dim1 by dim1 confusion Matrix, do the multiclass Matthews correlation coefficient
def matthews_corrcoef2(confusionMatrix):
    (dim1,dim2)=confusionMatrix.shape
    if (dim1 != dim2):
        raise ValueError("Confusion Matrix must be square")
    numerator = 0
    denominator1 = 0
    denominator2 = 0
    for k in range(dim1):
        for l in range(dim1):
            for m in range(dim1):
                numerator+=confusionMatrix[k,k]*confusionMatrix[l,m]
                numerator-=confusionMatrix[k,l]*confusionMatrix[m,k]
    for k in range(dim1):
        temp1=0
        temp2=0
        for k2 in range(dim1):
            if (k2 != k):
                for l2 in range(dim1):
                    temp1+=confusionMatrix[k2,l2]
                    temp2+=confusionMatrix[l2,k2]
        denominator1+=sum(confusionMatrix[k,:])*temp1
        denominator2+=sum(confusionMatrix[:,k])*temp2
    return numerator/(np.sqrt(denominator1*denominator2))

# given a dataframe rawdata, the number of policies and categories and functions
# that send each category to a policy and policies to natural numbers
# find the confusion matrix by iterating through the rows of df and adding 1 at a time
def findConfusionMatrix(rawData,numPolicies,numCategories,categoryToPolicy,policyToInt):
    confusionMatrix=np.zeros((numPolicies,numPolicies))
    for index, row in rawData.iterrows():
        policyAssigned=policyToInt(categoryToPolicy(row["Category"]))-1
        if (policyAssigned>=0 and policyAssigned<numPolicies):
            confusionMatrix[policyAssigned,policyToInt(row["Policy"])-1]+=1
        else:
            #print('currently unassigned so ignoring this row')
    return confusionMatrix

# same as above but passing dictionaries instead of functions for last two arguments
def findConfusionMatrix2(rawData,numPolicies,numCategories,categoryToPolicy,policyToInt):
    confusionMatrix=np.zeros((numPolicies,numPolicies))
    for index, row in rawData.iterrows():
        policyAssigned=policyToInt[categoryToPolicy[row["Category"]]]-1
        if (policyAssigned>=0 and policyAssigned<numPolicies):
            confusionMatrix[policyAssigned,policyToInt[row["Policy"]]-1]+=1
    return confusionMatrix

# automatically find the dictionary between categories to policies
# by going through one at a time at putting category k with policy p
# which gives the greatest MCC one at a time
# not really the best assignment to maximize MCC, but best one with all previous
# categories fixed and all later ones unassigned
def findMultiMCC(rawData):
    policies=df['Policy'].unique()
    numP = len(policies)
    categories=df['Category'].unique()
    numK = len(categories)
    count=0
    policyIntDict={}
    categoryPolicyDict={}
    categoryPolicyDict2={}
    for p in policies:
        policyIntDict[p]=count
        count+=1
    policyIntDict['unassigned']=-1
    for k in categories:
        categoryPolicyDict[k]='unassigned'
    for k in categories:
        best = - 1000000000
        categoryPolicyDict2=categoryPolicyDict.copy()
        for p in policies:
            categoryPolicyDict2[k]=p
            current=matthews_corrcoef2(np.ones((numP,numP))*.0000001+findConfusionMatrix2(rawData,numP,numK,categoryPolicyDict2,policyIntDict))
            print(categoryPolicyDict2)
            print(current)
            if (current>=best):
                best=current
                categoryPolicyDict[k]=p
    finalConfusionMatrix=findConfusionMatrix2(rawData,numP,numK,categoryPolicyDict,policyIntDict)
    finalMCC=matthews_corrcoef2(np.ones((numP,numP))*.0000001+finalConfusionMatrix)
    return (policyIntDict,categoryPolicyDict,finalConfusionMatrix,finalMCC)

def sampleCatToPolicy(category):
    if (category=='A'):
        return 1
    elif (category=='B'):
        return 2
    elif (category=='C'):
        return 4
    elif (category=='D'):
        return 1
    elif (category=='E'):
        return 3
    elif (category=='F'):
        return -1
    else:
        return -1

df = pd.read_csv('FairnessMCC/testCase.csv')
(policyIntDict,categoryPolicyDict,finalConfusionMatrix,finalMCC)=findMultiMCC(df)