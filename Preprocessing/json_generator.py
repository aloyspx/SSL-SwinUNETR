import argparse
import os
import random
import json
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_files_in_directory(directory):
    return sorted(os.listdir(directory))


def generate_ssl_json(args):
    set_seed(42)

    ct_files = load_files_in_directory(args.path)

    cut_index = int(args.ratio * len(ct_files))

    random.shuffle(ct_files)

    ct_files_train = ct_files[cut_index:]
    ct_files_val = ct_files[:cut_index]

    print(
        f"# Training samples: {len(ct_files_train)}\t# Validation samples: {len(ct_files_val)}"
    )

    training = []
    validation = []

    for ct in ct_files_train:
        training.append({"image": [f"./CT/{ct}"]})
    for ct in ct_files_val:
        validation.append({"image": [f"./CT/{ct}"]})

    data = {"training": training, "validation": validation}

    with open(args.json, "w") as f:
        json.dump(data, f)


def generate_supervised_json(args):
    set_seed(42)

    ct_files = load_files_in_directory(os.path.join(args.path, "Image"))
    mask_files = load_files_in_directory(os.path.join(args.path, "Mask"))

    assert len(ct_files) == len(mask_files)

    zipped = list(zip(ct_files, mask_files))
    random.shuffle(zipped)
    ct_files, mask_files = zip(*zipped)

    folds_ct = np.array_split(ct_files, args.folds, axis=0)
    folds_mask = np.array_split(mask_files, args.folds, axis=0)

    for i in range(args.folds):
        ct_files_val, mask_files_val = folds_ct[i], folds_mask[i]
        ct_files_train = np.concatenate(folds_ct[:i] + folds_ct[i + 1:], axis=0)
        mask_files_train = np.concatenate(folds_mask[:i] + folds_mask[i + 1:], axis=0)

        print(
            f"# Training samples: {len(ct_files_train)}\t# Validation samples: {len(ct_files_val)}"
        )

        training = []
        validation = []

        for ct, mask in zip(ct_files_train, mask_files_train):
            if not (ct.replace("CT", "") == mask.replace("Mask", "")):
                print(f"not aligned:\tct:{ct}\tmask:{mask}")

            training.append({
                "image": [f"./CT/{ct}"],
                "label": f"./Mask/{mask}"
            })

        for ct, mask in zip(ct_files_val, mask_files_val):
            assert ct.replace("CT", "") == mask.replace("SegP", "")
            validation.append({
                "image": [f"./CT/{ct}"],
                "label": f"./Mask/{mask}"
            })

        data = {"training": training, "validation": validation}

        with open(f"fold{i}.json", "w") as f:
            json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON for a dataset")
    parser.add_argument(
        "--path", default="dataset/dataset0", type=str, help="Path to the images"
    )
    parser.add_argument(
        "--json", default="jsons/dataset0.json", type=str, help="Path to the JSON output"
    )
    parser.add_argument(
        "--mode", type=str, help="Specify the data generation mode (ssl or sl)"
    )
    parser.add_argument(
        "--ratio", default=0.1, type=float, help="Ratio of validation data"
    )
    parser.add_argument(
        "--folds", default=1, type=int, help="Number of folds"
    )
    args = parser.parse_args()

    if args.mode == "ssl":
        generate_ssl_json(args)
    elif args.mode == "sl":
        generate_supervised_json(args)
