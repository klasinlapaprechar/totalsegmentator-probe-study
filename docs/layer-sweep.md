# Layer sweep

Embeddings pooled from six U-Net encoder stages of frozen `vertebrae_mr`. Layer retained by mean k-fold balanced accuracy on the training partition before locking the external evaluation. Early layers tended to win the binary contrast task in the published comparison block (`layer_01`).
