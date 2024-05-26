from compsoc.profile import Profile
from compsoc.evaluate import get_rule_utility
from compsoc.voting_rules.greedyN import greedyN_rule
from typing import Callable, Tuple

"""
ranking when topN = 1...
Calculate utility if topN = 1, topN =2, etc... 
Find average utility for that ranking
Repeat for when topN=2, etc.
Take the max

"""
def prove_greedyN(profile: Profile, vote: Tuple[int], verbose: bool = False):
    #calculate the total number of candidates
    num_cand = len(vote)
    topNlist = []

    #but ranking won't depend on topN 
    ranking = profile.ranking(greedyN_rule)

    greedyN_rule.__name__="greedyN"
    for N in range(1, num_cand//2):  
        topN = get_rule_utility(profile, greedyN_rule, N, verbose)
        topNlist.append(topN['topn'])
        
    
    optimum_u = max(topNlist)
    ave_util = sum(topNlist) / len(topNlist)

    return optimum_u, ave_util  


