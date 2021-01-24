import argparse
import features
import model as M
import numpy as np
import torch

from serialize import NNUEReader, NNUEWriter


def perform_surgery(model):
    # Perform surgery to avoid L1 weights clipping.
    l1_weights = model.l1.weight.data.numpy()
    max_abs_weight = 127/64

    x, y = np.where(np.abs(l1_weights) > max_abs_weight)
    input_neurons_to_rescale = []
    input_neurons_scale_factor = {}
    for i, j in zip(x, y):
        k = j % 256
        print(i, j, k)

        if k not in input_neurons_to_rescale:
            input_neurons_to_rescale.append(k)
            input_neurons_scale_factor[k] = 0.0

        input_neurons_scale_factor[k] = max(
            input_neurons_scale_factor[k], abs(l1_weights[i, j]))

    print(input_neurons_to_rescale)
    print(input_neurons_scale_factor)

    for k in input_neurons_to_rescale:
        # Rescale (reduce) L1 weights.
        model.l1.weight.data[:, k] *= 1/input_neurons_scale_factor[k]
        model.l1.weight.data[:, 256+k] *= 1/input_neurons_scale_factor[k]

        # Rescale (increase) corresponding input weights/biases.
        model.input.weight.data[k, :] *= input_neurons_scale_factor[k]
        model.input.bias.data[k] *= input_neurons_scale_factor[k]


def main():
    parser = argparse.ArgumentParser(
        description="Perform surgery on ckpt (or .pt) file.")
    parser.add_argument(
        "source", help="Source file (can be .ckpt, .pt)")
    parser.add_argument("target", help="Target file (can be .pt or .nnue)")
    features.add_argparse_args(parser)
    args = parser.parse_args()

    feature_set = features.get_feature_set_from_name(args.features)

    print('Performing surgery on %s and saving to %s' %
          (args.source, args.target))

    if args.source.endswith(".pt") or args.source.endswith(".ckpt"):
        if args.source.endswith(".pt"):
            nnue = torch.load(args.source)
        else:
            nnue = M.NNUE.load_from_checkpoint(
                args.source, feature_set=feature_set)
        nnue.eval()

        perform_surgery(nnue)

        if args.target.endswith(".pt"):
            torch.save(nnue, args.target)
        elif args.target.endswith(".nnue"):
            writer = NNUEWriter(nnue)
            with open(args.target, 'wb') as f:
                f.write(writer.buf)
        else:
            raise Exception('Invalid filetypes: ' + str(args))

    else:
        raise Exception('Invalid filetypes: ' + str(args))


if __name__ == '__main__':
    main()
