
from compsoc.profile import Profile

def findNConsensus(profile: Profile, candidate: int) ->int:

    #
    score = 0
    #num_voters = sum(votes for votes, _ in profile.ranking)
    
    #1-consensus
    for votes, ranking in profile.pairs:
        if ranking[0] == candidate:
            score += votes * 3 

    #2-consensus
    for votes, ranking in profile.pairs:
        if candidate in ranking[:2]:
            score += votes * 2  

    #3-consensus
    for votes, ranking in profile.pairs:
        if candidate in ranking[:3]:
            score += votes 

    last_place_votes = sum(votes for votes, ranking in profile.pairs if ranking[-1] == candidate)
    score -= last_place_votes 

    return score