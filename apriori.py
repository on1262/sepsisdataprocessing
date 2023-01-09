"""
Description     : Simple Python implementation of the Apriori Algorithm

Usage:
    $python apriori.py -f DATASET.csv -s minSupport  -c minConfidence

    $python apriori.py -f DATASET.csv -s 0.15 -c 0.6
"""

import sys
import pandas as pd
import numpy as np
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser



def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)

    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction): # example: {a1,b1} in row {a1,b1,c1}
                freqSet[item] += 1 # global frequency set
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet


def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )


def getItemSetTransactionList(df:pd.DataFrame):
    transactionList = list()
    itemSet = set()
    df = df.reset_index(drop=True)
    for _, record in df.iterrows():
        # 第一位是index, 不需要
        transaction = frozenset(record.values)
        transactionList.append(transaction)
        for item in transaction:
            if u'NAN' not in item:
                itemSet.add(frozenset([item]))  # Generate 1-itemSets
    return itemSet, transactionList


def runApriori(df:pd.DataFrame, consequents, max_iter=4, minSupport=0.1, minConfidence=0.5):
    """
    run the apriori algorithm. data_iter is a record iterator
    """
    consequents = [frozenset(c) for c in consequents]
    itemSet, transactionList = getItemSetTransactionList(df)
    print(f"Apriori: detected item={len(itemSet)}, transcation={len(transactionList)}")
    freqSet = defaultdict(int) # 记录所有项对应的出现次数
    largeSet = dict() # largeset[k]存储k组合的项集合, k>=1
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    assocRules = dict()
    # Dictionary which stores Association Rules

    oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)

    currentLSet = oneCSet
    k = 2
    while currentLSet != set([]) and k-1 <= max_iter: # 一直组合到没有满足min_support的结论
        print(f'Apriori: K={k-1}, set size={len(currentLSet)}')
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k) # 基于k-1项集, 制作长度为k的全部组合项集
        # 相消情况: [ab, bc]能组合成[abc]
        # 这里缺少一个prune操作, 如果ac不在频繁集里面, 那么任何含有ac的项都不需要计算support, 但对结果无影响, 对性能有损失
        currentCSet = returnItemsWithMinSupport(
            currentLSet, transactionList, minSupport, freqSet
        )
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)

    toRetItems = [] # 所有可行项以及它们的support
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    toRetRules = []
    for key, value in list(largeSet.items())[1:]: # 对于k>=2才开始算
        for item in value: # 迭代第k项集
            for remain in consequents: # 后件
                element = item.difference(remain) # 前件
                if len(element) > 0 and len(element) < len(item):
                    confidence = getSupport(item) / getSupport(element) # element能够推出remain的置信度
                    support = getSupport(item)
                    lift = getSupport(item) / (getSupport(element) * getSupport(remain))
                    # element=前件, remain=后件
                    if confidence >= minConfidence and lift > 1.0:
                        toRetRules.append(((tuple(element), tuple(remain)), confidence, support, lift))
    # - items (tuple, support)
    # - rules ((pretuple, posttuple), confidence, support, lift)
    items = [[(' | '.join(it[0])), it[1]] for it in toRetItems]
    item_df = pd.DataFrame(data=items, columns=[u'项', u'出现频率'])
    rules = [[(' | '.join(it[0][0])), ' | '.join(it[0][1]), it[1], it[2], it[3]] for it in toRetRules]
    rules_df = pd.DataFrame(data=rules, columns=[u'前件A', u'后件B', u'P(AB|A)', u'P(AB)', u'P(AB)/(P(A)P(B))'])
    return item_df, rules_df
