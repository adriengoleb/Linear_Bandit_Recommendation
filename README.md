# Linear-bandit- Academic Project (IASD)

The major problem of The Collaborative filtering technique for recommendations is : **the cold start**. Cold start happens when new users or new items arrive in e-commerce platforms. we have no browse, click or purchase data for them.

To solve this problem, given the fact that no prior information on the new user exists, one idea could be a random recommendation of movie to users. 
The idea can be better now if we can control the randomness in the exploration of different movie while keeping the exploitation of the most popular movie at a time. This is the usual dilemma between **Exploitation versus Exploration**.

This dilemma has been studied for decades in the multi-armed bandit framework. The goal of the player is to repeatedly pull the arms to maximize the expected total reward. He needs to explore different arms and at the same time exploit the current optimal arm.

In this repository, a project attempts to tackle the cold-start user problem by recommending a movie that will appeal to the user in question, rather than the best movie. We approach this by applying **non linear and Linear/contextual Bandits Algorithms** on the Data Base **MovieLens**.

