import argparse
import features
import model as M
import numpy as np
import torch
import matplotlib.pyplot as plt

from serialize import NNUEReader
from visualize import NNUEVisualizer

def correlate(models, net_names):
    # Coalesce input weights.
    weights_ = []

    for model in models:
        weight = model.input.weight.data
        indices = model.feature_set.get_virtual_to_real_features_gather_indices()
        weight_coalesced = weight.new_zeros(
            (weight.shape[0], model.feature_set.num_real_features))
        for i_real, is_virtual in enumerate(indices):
            weight_coalesced[:, i_real] = sum(
                weight[:, i_virtual] for i_virtual in is_virtual)

        weights_.append(weight_coalesced.transpose(0, 1).flatten().numpy())
    
    # Create mask to ignore weights for "Shogi piece drop" and pawns on first/last rank.
    numf = 256 # Number of features.
    numw_ = weights_[0].size // numf # Number of weights per feature.
    mask = []

    for i in range(numw_):
            # Calculate piece and king placement.
            pi = (i - 1) % 641
            ki = (i - 1) // 641
            piece = pi // 64
            rank = (pi % 64) // 8

            if pi == 640 or ((rank == 0 or rank == 7) and (piece == 0 or piece == 1)):
                # Ignore unused weights for "Shogi piece drop" and pawns on first/last rank.
                continue

            mask.append(i)

    mask = np.array(mask, dtype=int)
    
    # Mask weights.
    weights = []

    for w in weights_:
        weights.append(np.array([]))

        for j in range(numf):
            weights[-1] = np.concatenate([weights[-1], w[numf*mask + j]])

    numw = weights[0].size // numf

    # Transform to feature representations.
    if True:
        for (w, net_name) in zip(weights, net_names):
            numk = 64 # Number of king positions.        
            numwk = w.size // numk

            R = np.corrcoef(w.reshape((numf, numw)).transpose().reshape((numk, numwk)))
            plt.matshow(R)
            plt.title("Correlation coefficient w.r.t. king index [{}]".format(net_name))
            plt.colorbar()
    
    line_options = {'color': 'red', 'linewidth': 1}

    R = np.corrcoef(np.vstack([weights[0].reshape((numf, numw)), weights[1].reshape((numf, numw))]))
    plt.matshow(R)
    plt.title("Cross-correlation coefficient between features [{}, {}]".format(*net_names))
    plt.axhline(y=numf-0.5, **line_options)
    plt.axvline(x=numf-0.5, **line_options)
    plt.colorbar()
    
    # Try to reorder weights to maximize the cross-correlation.
    # Do this in a suboptimal heuristic way.
    remaining_features_first = list(range(numf))
    remaining_features_second = list(range(numf))
    reorder_ind_first = []
    reorder_ind_second = []

    C = R[:numf, numf:2*numf]
    while len(remaining_features_first) > 0:
        # Select the remaining feature pair (i,j) that has the max. cross-correlation.
        slice = C[remaining_features_first][:,remaining_features_second]
        i, j = np.unravel_index(np.argmax(slice), slice.shape)
        abs_i, abs_j = remaining_features_first[i], remaining_features_second[j]

        remaining_features_first.remove(abs_i)
        remaining_features_second.remove(abs_j)

        #print(C[abs_i][abs_j])

        reorder_ind_first.append(abs_i)
        reorder_ind_second.append(abs_j)
    
    inv_reorder_ind_first = np.zeros(numf, dtype=int)
    inv_reorder_ind_second = np.zeros(numf, dtype=int)
    for i in range(numf):
        inv_reorder_ind_first[reorder_ind_first[i]] = i
        inv_reorder_ind_second[reorder_ind_second[i]] = i

    R = np.corrcoef(np.vstack([weights[0].reshape((numf, numw))[reorder_ind_first], weights[1].reshape((numf, numw))[reorder_ind_second]]))
    plt.matshow(R)
    plt.title("Cross-correlation coefficient between reordered features [{}, {}]".format(*net_names))
    plt.axhline(y=numf-0.5, **line_options)
    plt.axvline(x=numf-0.5, **line_options)
    plt.colorbar()

    for i in range(2):
        visualizer = NNUEVisualizer(models[i])
        visualizer.plot_input_weights(
            net_names[i], 0, 1, save_dir=None, reorder=inv_reorder_ind_first if i==0 else inv_reorder_ind_second)

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Correlates networks in ckpt, pt and nnue format.")
    parser.add_argument(
        "first", help="First file (can be .ckpt, .pt or .nnue)")
    parser.add_argument(
        "second", help="Second file (can be .ckpt, .pt or .nnue)")
    features.add_argparse_args(parser)
    args = parser.parse_args()

    assert args.features == 'HalfKP'
    feature_set = features.get_feature_set_from_name(args.features)

    print("Correlating {} and {}".format(args.first, args.second))

    nnues = []
    for source in [args.first, args.second]:
        if source.endswith(".pt") or source.endswith(".ckpt"):
            if source.endswith(".pt"):
                nnue = torch.load(source)
            else:
                nnue = M.NNUE.load_from_checkpoint(
                    source, feature_set=feature_set)
            nnue.eval()
        elif source.endswith(".nnue"):
            with open(source, 'rb') as f:
                reader = NNUEReader(f, feature_set)
            nnue = reader.model
        else:
            raise Exception("Invalid filetype: " + str(args))
        
        nnues.append(nnue)

    from os.path import basename
    correlate(nnues, [basename(args.first), basename(args.second)])


if __name__ == '__main__':
    main()
