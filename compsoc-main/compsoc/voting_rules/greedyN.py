"""
Computes the score for a candidate.
"""
from compsoc.profile import Profile
import itertools

def greedyN_rule(profile: Profile, candidate: int) -> int:
    topN = 1


    """
    Computes the scores for candidates using a simple greedy, brute force rule that considers each ballot's top N candidates.


    :param profile: The voting profile.
    :type profile: VotingProfile
    :param candidate: The base candidate for scoring.
    :type candidate: int
    :return: The greedyN score for the candidate.
    :rtype: int
    """
    def scoreOne(profile: Profile, candidate: int) -> int:
        scores = 0
        for pair in profile.pairs:
            if candidate in pair[1]:
                for i in range(topN):
                    if pair[1].index(candidate) <= i:
                        #len(profile.candidates) / 2:
                        scores += pair[0]
        return scores
    
    def utilityForRanking (ranking: list[int], topN_parm: int) -> int:
        # Calculate utility of a given ranking for a given topN_parm value
        cand_ranking = ranking



        # iterate over the voters
        overall_utility = 0.0
        for pair in profile.pairs:
            # Utility of the ballot multipled by its counts
            num_candidates = len(profile.candidates)
            utility_increments = [(num_candidates - i) / (num_candidates * 1.0) for i in range(num_candidates)]
            total_utility = 0.0
            for i in range(min(topN_parm, len(pair[1]))):
                total_utility += utility_increments[cand_ranking.index(pair[1][i])]
            overall_utility += total_utility * pair[0]

        return overall_utility

    def compute_permutations(candidates):
        # base cases
        if len(candidates) == 0:
            return []
        elif len(candidates) == 1:
            return [candidates]
        
        permutations = []
        for candidate in range(len(candidates)):
            fir_cand = candidates[candidate]
            rest_cand = candidates[:candidate] + candidates[candidate + 1:]
            for rest in compute_permutations(rest_cand):
                permutations.append([fir_cand] + rest)
        return permutations

    # Compute optimal when topN is each possible value to find live options
    # Find whole ranking for topN=1, topN=2, etc    
    liveOptions = []
    if(len(profile.candidates)<=5):
        liveOptions = compute_permutations(list(profile.candidates))

    else:
        for j in range(len(profile.candidates) // 2):
            topN = j + 1
            rankings = profile.ranking(scoreOne)
            liveOptions.append(rankings)
        cand_ranking = []
        for pairs in liveOptions:
            temp =[]
            for j in pairs:
                temp.append(j[0])
            cand_ranking.append(temp)
        liveOptions = cand_ranking

    
    # Figure out which of the live options is the best expected value

    expectedUtilities = [0]* (len(liveOptions))
    # iterate over the live options
    for j in range(len(liveOptions)):
        # iterate over possible values of topN
        for m in range(len(profile.candidates) // 2):
            expectedUtilities[j] += utilityForRanking(liveOptions[j], m+1)


    # Find highest utility
    bestLiveOption = expectedUtilities.index(max(expectedUtilities))

    # Return the score for this candidate based on that ordering
    #topN = bestLiveOption + 1
    print(liveOptions)
    cand_ranking = liveOptions[bestLiveOption].index(candidate)
    
    return ((len(profile.candidates)) - cand_ranking)
