import pathlib
import random

import pandas as pd
import torch.nn
from rdkit import Chem
from sklearn.metrics import roc_auc_score, average_precision_score

from captum_attributions import explain_with_ig
from mpnn import mol_to_torch, MoleculeODetector, MoleculeFDetector
from noise import NoisyNetwork
from xai_testers import PositiveExplainabilityMethodTester


class NoiseEvaluator:
    def __init__(self):
        pass

    def test(self, model_fn, smiles_list: list[str], noises: list[float]) -> pd.DataFrame:
        noise_models = [NoisyNetwork(model_fn(), noise_std=noise) for noise in noises]
        all_outputs = []

        for smile in smiles_list:
            mol = Chem.MolFromSmiles(smile)
            mol = Chem.RemoveAllHs(mol)

            outputs = [smile]

            for noise_model in noise_models:
                example_x, edge_index, example_e = mol_to_torch(mol)
                output = noise_model(example_x, example_e, edge_index)
                outputs.append(output.item())

            all_outputs.append(outputs)

        df = pd.DataFrame(all_outputs, columns=["SMILES"] + [f"noise_{noise}" for noise in noises])

        return df


class NoiseTester:
    def __init__(self):
        pass

    def test(self, model_fn, model_smarts: str, smiles_list: list[str], noises: list[float],
             ground_truth: list[int]) -> pd.DataFrame:
        evaluator = NoiseEvaluator()
        results = evaluator.test(model_fn=model_fn, smiles_list=smiles_list, noises=noises)

        aurocs = []
        aps = []
        ap_baselines = []

        for noise in noises:
            preds = results[f"noise_{noise}"].tolist()

            auroc = roc_auc_score(ground_truth, preds)
            aurocs.append(auroc)

            ap = average_precision_score(ground_truth, preds)
            aps.append(ap)

            ap_baseline = sum(ground_truth) / len(ground_truth)
            ap_baselines.append(ap_baseline)

        df_dict = {
            "noise": noises,
            "auroc": aurocs,
            "ap": aps,
            "ap_baseline": ap_baselines,
        }

        summary_df = pd.DataFrame(df_dict)

        return summary_df


def main():
    model_fn = MoleculeFDetector
    model = model_fn()

    df = pd.read_csv(f"../data/validation_datasets_small/{model.__class__.__name__}.csv")
    df["LABEL"] = df["ORIGIN"] == "POSITIVE"

    positives = df[df["LABEL"] == True]
    negatives = df[df["LABEL"] == False]

    positives = positives.sample(n=100, random_state=42, replace=True)
    negatives = negatives.sample(n=100, random_state=42, replace=True)

    df = pd.concat([positives, negatives]).reset_index(drop=True)

    smiles, labels = df["SMILES"].tolist(), df["LABEL"].astype(int).tolist()

    noises = [elem / 100 for elem in range(0, 101)]

    tester = NoiseTester()
    results_df = tester.test(model_fn=model_fn,
                             model_smarts="[CHD3]C(=O)[NH][NH]C(=O)[CH2][CH2][CH2][CH2][CH2][CH2][CH3]",
                             smiles_list=smiles, noises=noises,
                             ground_truth=labels)
    print(results_df)

    results_df.to_csv(f"noise_test_results_{model.__class__.__name__}.csv", index=False)


if __name__ == "__main__":
    main()
