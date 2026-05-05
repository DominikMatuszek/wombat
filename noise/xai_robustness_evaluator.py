import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from rdkit import Chem
from sklearn.metrics import roc_auc_score, average_precision_score

from xai_methods import IGAttributionMethod
from highlight_smarts import highlight_atoms_in_mol
from krfp_models import krfp_models
from mpnn import mol_to_torch, MoleculeCDetector, MoleculeDDetector, MoleculeEDetector, MoleculeFDetector, \
    visualize_activations, visualize_all_activations
from noise import NoisyNetwork


class XAIRobustnessEvaluator:
    def __init__(self, model_fn, model_smarts: str):
        self.model_fn = model_fn
        self.model_smarts = model_smarts
        self.smarts_mol = Chem.MolFromSmarts(model_smarts)
        self.threshold = 0.5

    def check_if_removing_anyone_from_coalition_breaks_noisy_model(self, mol: Chem.Mol, noise: float,
                                                                   seed: int) -> bool:
        model = self.model_fn()
        model = NoisyNetwork(model, noise_std=noise, seed=seed)

        wanted_atoms = set(highlight_atoms_in_mol(mol, self.model_smarts))

        example_x, edge_index, example_e = mol_to_torch(mol, add_hydrogen_ohe=True)

        for atom in wanted_atoms:
            mutated_x = example_x.clone()

            mutated_x[
                atom] = 0  # Makes the atom effectively dead to the model. It LITERALLY cannot light up due to the architecture.

            pred = model(mutated_x, example_e, edge_index)

            if pred.item() >= self.threshold:
                print("Pred is", pred.item(), "after removing atom", atom,
                      "which is in the coalition. This should not happen.")

                pred, activations = model(mutated_x, example_e, edge_index, dump_activations=True)

                acts = visualize_all_activations(mol, activations)

                fig, axs = plt.subplots(3, len(acts) // 3, figsize=(30, 30), squeeze=False)

                for ax in axs.flatten():
                    ax.axis('off')

                for img, ax in zip(acts, axs.flatten()):
                    ax.imshow(img)

                plt.tight_layout()
                plt.gcf().set_dpi(200)
                plt.show()

                return False

        return True

    def check_if_coalition_wins_for_noisy_model(self, mol: Chem.Mol, noise: float, seed: int) -> bool:
        model = self.model_fn()
        model = NoisyNetwork(model, noise_std=noise, seed=seed)

        wanted_atoms = set(highlight_atoms_in_mol(mol, self.model_smarts))

        example_x, edge_index, example_e = mol_to_torch(mol, add_hydrogen_ohe=True)

        for atom in range(mol.GetNumAtoms()):
            if atom not in wanted_atoms:
                example_x[
                    atom] = 0  # Makes the atom effectively dead to the model. It LITERALLY cannot light up due to the architecture.

        pred = model(example_x, example_e, edge_index)

        return pred.item() >= self.threshold

    def check_if_noisy_model_is_correct(self, mol: Chem.Mol, noise_level: float, seed: int) -> bool:
        model = self.model_fn()
        model = NoisyNetwork(model, noise_std=noise_level, seed=seed)

        example_x, edge_index, example_e = mol_to_torch(mol, add_hydrogen_ohe=True)

        pred = model(example_x, example_e, edge_index)

        return pred.item() >= self.threshold

    def evaluate_attribution_method(self, explainability_method: callable, mol: Chem.Mol, noise_level: float,
                                    seed: int) -> None | tuple[float, float, float]:
        mol = Chem.RemoveAllHs(mol)

        assert mol is not None, f"Invalid molecule: {Chem.MolToSmiles(mol)}"
        assert len(mol.GetSubstructMatch(
            self.smarts_mol)) != 0, f"Model SMARTS {self.model_smarts} not found in molecule {Chem.MolToSmiles(mol)}. Robustness checking is only performed for positive molecules."

        correct = self.check_if_noisy_model_is_correct(mol, noise_level=noise_level, seed=seed)
        coalition_wins = self.check_if_coalition_wins_for_noisy_model(mol, noise=noise_level, seed=seed)
        coalition_breaks = self.check_if_removing_anyone_from_coalition_breaks_noisy_model(mol, noise=noise_level,
                                                                                           seed=seed)

        if not correct or not coalition_wins or not coalition_breaks:
            print("Rejecting molecule due to robustness checks. Details:")
            print(f"Correct: {correct}, Coalition wins: {coalition_wins}, Coalition breaks: {coalition_breaks}")
            return None

        example_x, edge_index, example_e = mol_to_torch(mol, add_hydrogen_ohe=True)

        model = self.model_fn()
        model = NoisyNetwork(model, noise_std=noise_level, seed=seed)

        explainability_method_obj = explainability_method(
            model=model,
            model_smarts=self.model_smarts,
            positive_smiles=[],
            negative_smiles=[],
        )

        node_attrs, _ = explainability_method_obj.explain(
            example_x=example_x,
            example_e=example_e,
            edge_index=edge_index,
        )

        node_attrs = torch.sum(torch.abs(node_attrs), dim=1)
        wanted_atoms = set(highlight_atoms_in_mol(mol, self.model_smarts))

        prediction_mask = [1 if i in wanted_atoms else 0 for i in range(mol.GetNumAtoms())]

        prediction_mask = np.array(prediction_mask)
        node_attrs = node_attrs.squeeze().detach().numpy()

        auroc = roc_auc_score(prediction_mask, node_attrs)
        ap = average_precision_score(prediction_mask, node_attrs)
        ap_baseline = len(wanted_atoms) / mol.GetNumAtoms()

        return auroc, ap, ap_baseline


def main():
    df = pd.read_csv("../data/validation_datasets_small/MoleculeFDetector.csv")
    df = df[df["ORIGIN"] == "POSITIVE"]

    smiles = df["SMILES"].tolist()
    smiles = smiles[1]

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveAllHs(mol)

    evaluator = XAIRobustnessEvaluator(model_fn=MoleculeFDetector, model_smarts=krfp_models[2][2])

    for noise in [elem / 1000 for elem in range(0, 1001)]:
        for seed in range(5):
            result = evaluator.evaluate_attribution_method(explainability_method=IGAttributionMethod, mol=mol,
                                                           noise_level=noise, seed=seed)

            if result is not None:
                auroc, ap, ap_baseline = result
                print(
                    f"Noise: {noise:.2f}, Seed: {seed}, AUROC: {auroc:.4f}, AP: {ap:.4f}, AP Baseline: {ap_baseline:.4f}")
            else:
                print(f"Noise: {noise:.2f}, Seed: {seed}, Result: None (model failed robustness checks)")
                exit()


if __name__ == "__main__":
    main()
