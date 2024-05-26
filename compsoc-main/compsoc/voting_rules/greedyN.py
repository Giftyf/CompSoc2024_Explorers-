"""
Computes the greedy score for a candidate.
"""
from compsoc.profile import Profile

def greedyN_rule(profile: Profile, candidate: int) -> int:
    """
    Calculates the greedy score for a candidate based on a profile.

    :param profile: The voting profile.
    :type profile: VotingProfile
    :param candidate: The base candidate for scoring.
    :type candidate: int
    :return: The greedy score for the candidate.
    :rtype: int
    """
   
    # Get pairwise scores
    scores = 0
    for pair in profile.pairs:
        if candidate in pair[1]:
            for i in range(len(profile.candidates) // 2):
                if pair[1].index(candidate) <= i:
                    #len(profile.candidates) / 2:
                    scores += pair[0]
    return scores