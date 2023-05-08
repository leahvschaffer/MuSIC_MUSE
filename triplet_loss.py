'''
Triplet loss modified from https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
Author: Olivier Moindrot from Stanford
CONVERTED TO PYTORCH LVS
'''
import torch
import torch.nn as nn

def _pairwise_distances(embeddings, device):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    distances = 1 - torch.nn.functional.cosine_similarity(embeddings[:,:,None], embeddings.t()[None,:,:])

    return distances

def _get_triplet_mask(labels, device):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size()[0]).bool().to(device)
    indices_not_equal = torch.logical_not(indices_equal)
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = labels > 0
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)

    valid_labels = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

    # Combine the two masks
    mask = torch.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings, margin, device):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
    
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, device)
    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
    assert anchor_positive_dist.size()[2] == 1, "{}".format(anchor_positive_dist.size())
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)
    assert anchor_negative_dist.size()[1] == 1, "{}".format(anchor_negative_dist.size())

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels, device)
    mask = mask.float()
    triplet_loss = torch.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = torch.maximum(triplet_loss, torch.zeros(triplet_loss.size()).to(device))

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = torch.greater(triplet_loss, 1e-16).float()
    num_positive_triplets = torch.sum(valid_triplets)
    num_valid_triplets = torch.sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def fraction_triplets(labels, embeddings, margin, device):
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, device)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
    assert anchor_positive_dist.size()[2] == 1, "{}".format(anchor_positive_dist.size())
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)
    assert anchor_negative_dist.size()[1] == 1, "{}".format(anchor_negative_dist.size())

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels, device)
    mask = mask.float()
    triplet_loss = torch.multiply(mask, triplet_loss)

    easy_triplets = torch.less(triplet_loss, 0)
    semi_hard_triplets = torch.logical_and(torch.greater(triplet_loss, 0), torch.less(triplet_loss, margin))
    hard_triplets = torch.greater(triplet_loss, margin)
    
    num_easy_triplets = torch.sum(easy_triplets)
    num_semi_hard_triplets = torch.sum(semi_hard_triplets)
    num_hard_triplets = torch.sum(hard_triplets)
    
    num_valid_triplets = torch.sum(mask)
    fraction_easy_triplets = num_easy_triplets / (num_valid_triplets + 1e-16)
    fraction_semi_hard_triplets = num_semi_hard_triplets / (num_valid_triplets + 1e-16)
    fraction_hard_triplets = num_hard_triplets / (num_valid_triplets + 1e-16)
    
    return fraction_easy_triplets, fraction_semi_hard_triplets, fraction_hard_triplets


def batch_hard_triplet_loss(labels, embeddings, margin, device):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, device)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
    assert anchor_positive_dist.size()[2] == 1, "{}".format(anchor_positive_dist.size())
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)
    assert anchor_negative_dist.size()[1] == 1, "{}".format(anchor_negative_dist.size())

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels, device)
    mask = mask.float()
    triplet_loss = torch.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss[triplet_loss < 0] = 0

    # max per anchor
    maxes = torch.amax(triplet_loss,dim=(1,2))
    triplet_loss = torch.mean(maxes[mask.sum(dim=(1,2)) > 0]) 

    return triplet_loss