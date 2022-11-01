# graph-based-exploration-access

The code to reproduce "EXPLORATION IN SEQUENTIAL RECOMMENDER SYSTEMS VIA GRAPH REPRESENTATIONS"

The code is based on the original TGN implementation https://github.com/twitter-research/tgn.
The main contribution lies in the file [decision_module.py](modules/decision_module.py).
Also, we have implemented the [ope_loss_module.py](modules/ope_loss_module.py) with Replay counter-factual evaluation applied to the loss
and modified train loop in [train_ope_offline.py](train_ope_offline.py) to run an online simulation. 


To reproduce the paper run the [notebook](run-and-build-tables.ipynb).
