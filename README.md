# GPU-accelerated-agents-simulations
Repository for all my GPU simulations

### SEIIQR
This is a simulation of agents infecting each other based on a random probability and time spent next to each other.
It was implemented with a spatial hash grid which can run a simulation of 100,000 agents for 250 iterations in 3~ seconds, compared with the
naive approach for distance checking which took around ~63 seconds

A buffer is allocated upfront for all the agents across all the iterations to speed up the calculations

![](https://github.com/Pharadas/GPU-accelerated-agents-simulations/blob/master/SEIIQR_simulation/append_animation.gif)
