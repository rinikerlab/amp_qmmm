import torch
import numpy as np
import torchlayers as tl
import time
import yaml
import sys
import gc
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError as MAE

import torchlayers as tl
from AMPQMMM import AMPQMMM as AMPQMMM_Precision
from AMPQMMMmin import AMPQMMM as AMPQMMM_Minimal

H_TO_KJ = 627.509474 * 4.184
BOHR_TO_ANGSTROM = 0.529177210903
ENERGY_CONVERSION = H_TO_KJ
MULTIPOLE_CONVERSION = BOHR_TO_ANGSTROM
FORCE_CONVERSION = H_TO_KJ / BOHR_TO_ANGSTROM


class SingleSystemOrcaXtbDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        system_name: str,
        start_idx: int,
        end_idx: int,
        last_idx: int,
        dtype: str,
        delta_qmmm: bool,
        delta_qm: bool,
        multi_loss: bool,
        load_delta: bool,
    ):
        assert (end_idx is None) or (end_idx > start_idx)
        # save member variables
        self._data_path = data_path
        self._system_name = system_name
        self._delta_qmmm = delta_qmmm
        self._delta_qm = delta_qm
        self._multi_loss = multi_loss
        self._load_delta = load_delta
        self._dtype = dtype
        self._arrays = dict()
        print(f"Trying to open {data_path}, {system_name}")
        try:
            print("Trying to read input data from Orca results...")
            self._qm_coordinates = np.load(
                f"{data_path}/{system_name}/orca_coordinates.npy",
                allow_pickle=True,
            )
            assert len(self._qm_coordinates) == last_idx
            self._mm_coordinates = np.load(
                f"{data_path}/{system_name}/orca_pc_coordinates.npy",
                allow_pickle=True,
            )
            assert len(self._mm_coordinates) == last_idx
            self._qm_charges = np.load(
                f"{data_path}/{system_name}/orca_species.npy",
                allow_pickle=True,
            )
            assert len(self._qm_charges) == last_idx
            self._mm_charges = np.load(
                f"{data_path}/{system_name}/orca_pc_charges.npy",
                allow_pickle=True,
            )
        except:
            print("Falling back to xtb results...")
            self._qm_coordinates = np.load(
                f"{data_path}/{system_name}/xtb_coordinates.npy",
                allow_pickle=True,
            )
            self._qm_coordinates = self._qm_coordinates * BOHR_TO_ANGSTROM
            assert len(self._qm_coordinates) == last_idx
            self._mm_coordinates = np.load(
                f"{data_path}/{system_name}/xtb_pc_coordinates.npy",
                allow_pickle=True,
            )
            self._mm_coordinates = self._mm_coordinates * BOHR_TO_ANGSTROM
            assert len(self._mm_coordinates) == last_idx
            self._qm_charges = np.load(
                f"{data_path}/{system_name}/xtb_species.npy",
                allow_pickle=True,
            )
            assert len(self._qm_charges) == last_idx
            self._mm_charges = np.load(
                f"{data_path}/{system_name}/xtb_pc_charges.npy",
                allow_pickle=True,
            )
            assert len(self._mm_charges) == last_idx
        # move away empty (padded) MM particles
        self._mm_coordinates[np.where(np.abs(self._mm_coordinates).sum(-1) == 0)] += 1e5
        self._qm_energies = np.load(
            f"{data_path}/{system_name}/orca_energies.npy",
            allow_pickle=True,
        )
        assert len(self._qm_energies) == last_idx
        self._qm_gradients = np.load(
            f"{data_path}/{system_name}/orca_engrad.npy",
            allow_pickle=True,
        )
        assert len(self._qm_gradients) == last_idx
        self._mm_gradients = np.load(
            f"{data_path}/{system_name}/orca_pcgrad.npy",
            allow_pickle=True,
        )
        assert len(self._mm_gradients) == last_idx
        # save e0 (global)
        self._e0_idx = np.argmin(np.abs(self._qm_energies - np.mean(self._qm_energies)))
        self._e0 = self._qm_energies[self._e0_idx]
        # subtract e0
        self._qm_energies = self._qm_energies - self._e0
        # take the correct slices
        self._qm_coordinates = self._qm_coordinates[start_idx:end_idx]
        self._mm_coordinates = self._mm_coordinates[start_idx:end_idx]
        self._qm_energies = self._qm_energies[start_idx:end_idx]
        self._qm_gradients = self._qm_gradients[start_idx:end_idx]
        self._mm_gradients = self._mm_gradients[start_idx:end_idx]
        self._qm_charges = self._qm_charges[start_idx:end_idx]
        self._mm_charges = self._mm_charges[start_idx:end_idx]
        self._qm_energies = (
            self._qm_energies.reshape([-1, 1]) * ENERGY_CONVERSION
        )
        self._qm_gradients = (self._qm_gradients * FORCE_CONVERSION)
        self._mm_gradients = (self._mm_gradients * FORCE_CONVERSION)
        self._qm_charges = self._qm_charges.astype(np.int64)
        self._mm_charges = self._mm_charges
        # save available data
        self._arrays["qm_coordinates"] = self._qm_coordinates
        self._arrays["mm_coordinates"] = self._mm_coordinates
        self._arrays["qm_energies"] = self._qm_energies
        self._arrays["qm_gradients"] = self._qm_gradients
        self._arrays["mm_gradients"] = self._mm_gradients
        self._arrays["qm_charges"] = self._qm_charges
        self._arrays["mm_charges"] = self._mm_charges
        # delta model on qm energies, qm gradients, and mm gradients
        if delta_qmmm or load_delta:
            self._delta_qm_energies = np.load(
                f"{data_path}/{system_name}/xtb_energies.npy",
                allow_pickle=True,
            )
            assert len(self._delta_qm_energies) == last_idx
            self._delta_qm_gradients = np.load(
                f"{data_path}/{system_name}/xtb_engrad.npy",
                allow_pickle=True,
            )
            assert len(self._delta_qm_gradients) == last_idx
            self._delta_mm_gradients = np.load(
                f"{data_path}/{system_name}/xtb_pcgrad.npy",
                allow_pickle=True,
            )
            assert len(self._delta_mm_gradients) == last_idx
            # save de0 (global) in dataset
            self._de0 = self._delta_qm_energies[self._e0_idx]
            self._delta_qm_energies = self._delta_qm_energies - self._de0
            # take the correct slices
            self._delta_qm_energies = self._delta_qm_energies[start_idx:end_idx]
            self._delta_qm_gradients = self._delta_qm_gradients[start_idx:end_idx]
            self._delta_mm_gradients = self._delta_mm_gradients[start_idx:end_idx]
            # convert to correct units
            self._delta_qm_energies = (
                self._delta_qm_energies.reshape([-1, 1]) * ENERGY_CONVERSION
            )
            self._delta_qm_gradients = (
                self._delta_qm_gradients * FORCE_CONVERSION
            )
            self._delta_mm_gradients = (
                self._delta_mm_gradients * FORCE_CONVERSION
            )
            # save available data
            self._arrays["delta_qm_energies"] = self._delta_qm_energies
            self._arrays["delta_qm_gradients"] = self._delta_qm_gradients
            self._arrays["delta_mm_gradients"] = self._delta_mm_gradients
        # delta model when xtb coordinates were re-evaluated in absence of MM pointcharges
        elif delta_qm:
            self._delta_qm_energies = np.load(
                f"{data_path}/{system_name}/xtb_energies_reeval.npy",
                allow_pickle=True,
            )
            assert len(self._delta_qm_energies) == last_idx
            self._delta_qm_gradients = np.load(
                f"{data_path}/{system_name}/xtb_engrad_reeval.npy",
                allow_pickle=True,
            )
            assert len(self._delta_qm_gradients) == last_idx
            # save de0 in dataset (identical for training, validation, and test set)
            self._de0 = self._delta_qm_energies[self._e0_idx]
            self._delta_qm_energies = self._delta_qm_energies - self._de0
            # take the correct slices
            self._delta_qm_energies = self._delta_qm_energies[start_idx:end_idx]
            self._delta_qm_gradients = self._delta_qm_gradients[start_idx:end_idx]
            # convert to correct units
            self._delta_qm_energies = (
                self._delta_qm_energies.reshape([-1, 1]) * ENERGY_CONVERSION
            )
            self._delta_qm_gradients = (self._delta_qm_gradients * FORCE_CONVERSION)
            # save available data
            self._self._arrays["delta_qm_energies"] = self._delta_qm_energies
            self._self._arrays["delta_qm_gradients"] = self._delta_qm_gradients
        # take loss with respect to dipoles and quadrupoles into account
        if multi_loss:
            self._qm_dipoles = np.load(
                f"{data_path}/{system_name}/orca_dipoles.npy",
                allow_pickle=True,
            )
            assert len(self._qm_dipoles) == last_idx
            self._qm_quadrupoles = np.load(
                f"{data_path}/{system_name}/orca_quadrupoles.npy",
                allow_pickle=True,
            )
            assert len(self._qm_quadrupoles) == last_idx
            # take the correct slices
            self._qm_dipoles = self._qm_dipoles[start_idx:end_idx]
            self._qm_quadrupoles = self._qm_quadrupoles[start_idx:end_idx]
            # convert to correct units
            self._qm_dipoles = (self._qm_dipoles * MULTIPOLE_CONVERSION)
            self._qm_quadrupoles[:, :3] -= np.mean(
                self._qm_quadrupoles[:, :3], axis=-1, keepdims=True
            )
            self._qm_quadrupoles = (
                self._qm_quadrupoles * (MULTIPOLE_CONVERSION**2)
            )
            # save available data
            self._arrays["qm_dipoles"] = self._qm_dipoles
            self._arrays["qm_quadrupoles"] = self._qm_quadrupoles
        # assert all tensor have equal length
        for value in self._arrays.values():
            assert(self._arrays["qm_energies"].shape[0] == value.shape[0])
        self._length = self._arrays["qm_energies"].shape[0]
        # cast dtype
        for key in self._arrays:
            if key != "qm_charges":
                if self._dtype == "float32":
                    self._arrays[key] = np.float32(self._arrays[key])
                elif self._dtype == "float64":
                    self._arrays[key] = np.float64(self._arrays[key])
                else:
                    print(f"Unsupported dtype: {self._dtype}")
                    sys.exit(1)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        assert idx < self._length
        arrays = dict()
        for key, array in self._arrays.items():
            arrays[key] = array[idx] 
        return arrays


class MultiSystemOrcaXtbDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        system_name: str,
        stage: str,
        dtype: str,
        delta_qmmm: bool,
        delta_qm: bool,
        multi_loss: bool,
    ):
        assert not delta_qm or not delta_qmmm, "Delta QM and delta QM/MM are currently not tested for multisystems."
        # save member variables
        self._data_path = data_path
        self._system_name = system_name
        self._stage = stage
        self._base_path = os.path.join(self._data_path, self._system_name, self._stage)
        self._delta_qmmm = delta_qmmm
        self._delta_qm = delta_qm
        self._multi_loss = multi_loss
        self._dtype = dtype
        self._batch_files = [os.path.join(self._base_path, file) for file in os.listdir(self._base_path) if "batch" in file and file.endswith(".npy")]
        # assume everything fits into main memory
        self._batch_directories = [np.load(file, allow_pickle=True).item() for file in self._batch_files]
        self._length = len(self._batch_files)
        for batch_directory in self._batch_directories:
            # assert same number of frames per dimension
            current_num_frames = batch_directory["qm_energies"].shape[0]
            for item in batch_directory.values():
                assert len(item) == current_num_frames
            # assert all entries in the current batch are from the same molecule
            assert np.all(np.all(batch_directory["qm_charges"] == batch_directory["qm_charges"][0, :], axis=1))
            batch_directory["delta_qm_energies"] = batch_directory["delta_qm_energies"] - batch_directory["delta_qm_energies"][0]
            batch_directory["delta_qm_energies"] = batch_directory["delta_qm_energies"].reshape([-1, 1]) * ENERGY_CONVERSION
            batch_directory["delta_qm_gradients"] = batch_directory["delta_qm_gradients"] * FORCE_CONVERSION
            batch_directory["delta_mm_gradients"] = batch_directory["delta_mm_gradients"] * FORCE_CONVERSION
            batch_directory["qm_charges"] = batch_directory["qm_charges"].astype(np.int64)
            # move away empty (padded) MM particles
            batch_directory["mm_coordinates"][np.where(np.abs(batch_directory["mm_coordinates"]).sum(-1) == 0)] += 1e5
            batch_directory["qm_energies"] = batch_directory["qm_energies"].reshape([-1, 1]) * ENERGY_CONVERSION
            batch_directory["qm_energies"] = batch_directory["qm_energies"] - batch_directory["qm_energies"][0]
            batch_directory["qm_gradients"] = batch_directory["qm_gradients"] * FORCE_CONVERSION
            batch_directory["mm_gradients"] = batch_directory["mm_gradients"] * FORCE_CONVERSION
            if self._delta_qmmm:
                # train model to predict difference between Orca energy and xtb energy, gradient, and pc gradient
                batch_directory["delta_qm_energies"] = batch_directory["qm_energies"] - batch_directory["delta_qm_energies"]
                batch_directory["delta_qm_gradients"] = batch_directory["qm_gradients"] - batch_directory["delta_qm_gradients"]
                batch_directory["delta_mm_gradients"] = batch_directory["mm_gradients"] - batch_directory["delta_mm_gradients"]
            elif self._delta_qm:
                # train model to predict difference between Orca energy and xtb energy, gradient, but full Orca pc gradient
                batch_directory["delta_qm_energies"] = batch_directory["qm_energies"] - batch_directory["delta_qm_energies"]
                batch_directory["delta_qm_gradients"] = batch_directory["qm_gradients"] - batch_directory["delta_qm_gradients"]
                batch_directory["delta_mm_gradients"] = batch_directory["mm_gradients"]
            else:
                # drop superfluous entries
                del batch_directory["delta_qm_energies"]
                del batch_directory["delta_qm_gradients"]
                del batch_directory["delta_mm_gradients"]
            if not self._multi_loss:
                # drop superfluous entries
                del batch_directory["qm_dipoles"]
                del batch_directory["qm_quadrupoles"]
            else:
                batch_directory["qm_dipoles"] = batch_directory["qm_dipoles"] * MULTIPOLE_CONVERSION
                batch_directory["qm_quadrupoles"][:, :3] = batch_directory["qm_quadrupoles"][:, :3] - np.mean(batch_directory["qm_quadrupoles"][:, :3], axis=-1, keepdims=True)
                batch_directory["qm_quadrupoles"] = batch_directory["qm_quadrupoles"] * (MULTIPOLE_CONVERSION ** 2)
            # cast dtype
            for key in batch_directory:
                if key != "qm_charges":
                    if self._dtype == "float32":
                        batch_directory[key] = np.float32(batch_directory[key])
                    elif self._dtype == "float64":
                        batch_directory[key] = np.float64(batch_directory[key])
                    else:
                        print(f"Unsupported dtype: {self._dtype}")
                        sys.exit(1)

    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        assert index < self._length
        batch = dict()
        for key, value in self._batch_directories[index].items():
            batch[key] = torch.as_tensor(value)
        return batch



INPUT_LAYOUT = {
    "qm_charges" : 0,
    "qm_coordinates" : 1,
    "mm_atoms" : 2,
    "mm_charges" : 3,
    "mm_coordinates" : 4,
}


def load_parameters_file(filename: str):
    file = open(filename, "r")
    PARAMETERS = yaml.load(file, yaml.Loader)
    PARAMETERS["time"] = int(time.time())

    return PARAMETERS

def batch_to_input(batch: tuple) -> tuple:
    # check if all atoms are identical (only one molecule)
    assert(all(torch.all(torch.eq(batch["qm_charges"][0], batch["qm_charges"][i])) for i in range(1, batch["qm_charges"].size(0))))
    qm_charges = batch["qm_charges"][0]
    qm_coordinates = batch["qm_coordinates"]
    mm_charges = batch["mm_charges"]
    mm_coordinates = batch["mm_coordinates"]
    return (qm_charges, qm_coordinates, None, mm_charges, mm_coordinates)


def instantiate_model(PARAMETERS: dict, training_data):
    # loader to instantiate model
    if PARAMETERS["single_system"]:
        instantiation_loader = DataLoader(training_data, batch_size=PARAMETERS["batch_size"], shuffle=False)
        # save E0 (and dE0)
        PARAMETERS["E0"] = training_data._e0.item()
        PARAMETERS["E0_IDX"] = training_data._e0_idx.item()
        if PARAMETERS["delta_qmmm"] or PARAMETERS["delta_qm"]:
            PARAMETERS["dE0"] = training_data._de0.item()
        print(f"E0 saved in parameters: {PARAMETERS['E0']}")
        print(f"E0_IDX saved in parameters: {PARAMETERS['E0_IDX']}")
        if PARAMETERS["delta_qmmm"] or PARAMETERS["delta_qm"]:
            print(f"dE0 saved in parameters: {PARAMETERS['dE0']}")
    else:
        instantiation_loader = DataLoader(training_data, collate_fn=lambda x: x[0], shuffle=False)

    # backup user selection
    user_device_name = PARAMETERS["device_name"]
    user_device = torch.device(user_device_name)
    cpu_name = "cpu"
    cpu_device = torch.device(cpu_name)

    if PARAMETERS["model_architecture"] == "precision":
        model = AMPQMMM_Precision(**PARAMETERS)
    elif PARAMETERS["model_architecture"] == "minimal":
        model = AMPQMMM_Minimal(**PARAMETERS)
    else:
        print(f"unknown model architecture: {PARAMETERS['model_architecture']}")
        sys.exit(1)

    sample_batch = next(iter(instantiation_loader))

    # check dtype
    for key in sample_batch:
        if sample_batch[key] is not None and sample_batch[key].dtype != torch.int64:
            if PARAMETERS["dtype"] == "float32":
                assert sample_batch[key].dtype == torch.float32
            elif PARAMETERS["dtype"] == "float64":
                assert sample_batch[key].dtype == torch.float64
            else:
                print(f"Unsupported dtype: {PARAMETERS['dtype']}")
                sys.exit(1)
            
    # set CPU
    model.device = cpu_device
    model.dtype = PARAMETERS["dtype"]

    sample_input = (sample_batch["qm_charges"][0], sample_batch["qm_coordinates"], None, sample_batch["mm_charges"], sample_batch["mm_coordinates"])

    model = tl.build(model, sample_input)
    print("Sample batch on CPU:")
    prediction_cpu, _ = model.forward_with_graph(sample_input)
    print("Prediction (CPU):")
    print(prediction_cpu.detach())

    if PARAMETERS["single_system"]:
        print("Prediction + E0 (CPU)")
        print(prediction_cpu.detach() + PARAMETERS["E0"])
        print()

    # move data back to GPU
    if user_device != cpu_device:
        model.to(user_device)
        model.device = user_device
        # sample input on device
        sample_input = [input.to(user_device) if isinstance(input, torch.Tensor) else None for input in sample_input]
        prediction_cuda, _ = model.forward_with_graph(sample_input)
        print("Prediction (Device):")
        print(prediction_cuda.detach())
        if PARAMETERS["single_system"]:
            print("Prediction + E0 (Device)")
            print(prediction_cuda.detach() + PARAMETERS["E0"])
            print()

    return model



def set_model_dtype(model, PARAMETERS):
    if PARAMETERS["dtype"] == "float32" or PARAMETERS["dtype"] == torch.float32:
        torch.set_default_dtype(torch.float32)
        model = model.to(dtype=torch.float32)
        model.dtype = torch.float32
    elif PARAMETERS["dtype"] == "float64" or PARAMETERS["dtype"] == torch.float64:
        torch.set_default_dtype(torch.float64)
        model = model.to(dtype=torch.float64)
        model.dtype = torch.float64
    else:
        print(f"Unsupported dtype: {PARAMETERS['dtype']}")
        sys.exit(1)
    
    return model

def load_state_dict(model, PARAMETERS):
    model.load_state_dict(torch.load(os.path.join(PARAMETERS["save_path"], f"{PARAMETERS['model_name']}_state_dict.pth")))
    assert_correct_dtype(model, PARAMETERS)

    return model


def log_general_stats(model_name, experiment_name, best_epoch, epochs_trained, epochs_scheduled, last_tmae, last_vmae):
    print(f"Model: {model_name}")
    print(f"Experiment: {experiment_name}")
    print(f"Best epoch: {best_epoch}")
    print(f"Epochs trained: {epochs_trained}")
    print(f"Epochs scheduled: {epochs_scheduled}")
    print(f"Last training MAE: {last_tmae}")
    print(f"Last validation MAE: {last_vmae}")


def evaluate_on_dataset(model, stage, data_loader, PARAMETERS):
    evaluation_start = time.time()

    mae_energy = MAE().to(model.device)
    mae_gradient_qm = MAE().to(model.device)
    mae_gradient_mm = MAE().to(model.device)
    if PARAMETERS['multi_loss']:
        mae_dipoles = MAE().to(model.device)
        mae_quadrupoles = MAE().to(model.device)

    # Set the model to evaluation mode
    model.eval()

    for idx, batch in enumerate(data_loader):
        if idx % 100 == 0:
            print(f"Testing stage: {stage}... Batch {idx} / {len(data_loader)}")

        # transfer batch to GPU and prepare input
        for key, tensor in batch.items():
            batch[key] = tensor.to(model.device) if isinstance(tensor, torch.Tensor) else None
            if key == "qm_coordinates" or key == "mm_coordinates":
                batch[key].requires_grad = True

        # check dtype
        for key in batch:
            if batch[key] is not None and batch[key].dtype != torch.int64:
                if PARAMETERS["dtype"] == "float32":
                    assert batch[key].dtype == torch.float32
                elif PARAMETERS["dtype"] == "float64":
                    assert batch[key].dtype == torch.float64
                else:
                    print(f"Unsupported dtype: {PARAMETERS['dtype']}")
                    sys.exit(1)

        # make prediction
        input = batch_to_input(batch)

        # make prediction
        prediction, graph = model.forward_with_graph(input)
        qm_gradients_pred = torch.autograd.grad(prediction, input[INPUT_LAYOUT["qm_coordinates"]], grad_outputs=torch.ones_like(prediction), retain_graph=True)[0]
        mm_gradients_pred = torch.autograd.grad(prediction, input[INPUT_LAYOUT["mm_coordinates"]], grad_outputs=torch.ones_like(prediction), retain_graph=False)[0]

        if not PARAMETERS["single_system"]:
            prediction = prediction - prediction[0]

        if PARAMETERS['delta_qm']:
            prediction += batch['delta_qm_energies']
            qm_gradients_pred += batch['delta_qm_gradients']
        elif PARAMETERS['delta_qmmm']:
            prediction += batch['delta_qm_energies']
            qm_gradients_pred += batch['delta_qm_gradients']
            mm_gradients_pred += batch['delta_mm_gradients']
        if PARAMETERS['multi_loss']:
            pred_dipole = model._molecular_dipole(graph)
            pred_quadrupole = model._molecular_quadrupole(graph)

        # compute MAE (validation)
        mae_energy.update(batch['qm_energies'], prediction)
        mae_gradient_qm.update(batch['qm_gradients'], qm_gradients_pred)
        mae_gradient_mm.update(batch['mm_gradients'], mm_gradients_pred)
        if PARAMETERS['multi_loss']:
            mae_dipoles.update(batch['qm_dipoles'], pred_dipole)
            mae_quadrupoles.update(batch['qm_quadrupoles'], pred_quadrupole[:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]])

    maes = dict()
    maes["mae_energies"] = mae_energy.compute()
    maes["mae_qm_gradients"] = mae_gradient_qm.compute()
    maes["mae_mm_gradients"] = mae_gradient_mm.compute()
    if PARAMETERS['multi_loss']:
        maes["mae_dipoles"] = mae_dipoles.compute()
        maes["mae_quadrupoles"] = mae_quadrupoles.compute()
    
    evaluation_end = time.time()
    evaluation_time = evaluation_end - evaluation_start

    return maes, evaluation_time

def dipole_loss(target_dipoles, pred_dipoles, loss_fn):
    return loss_fn(target_dipoles, pred_dipoles)


def quadrupole_loss(target_quadrupoles, pred_quadrupoles, loss_fn):
    return loss_fn(
        target_quadrupoles, pred_quadrupoles[:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    )


def assert_correct_dtype(model, PARAMETERS):
    # assert model has precision requested
    for param in model.parameters():
        if PARAMETERS["dtype"] == "float32":
            assert param.dtype == torch.float32
        elif PARAMETERS["dtype"] == "float64":
            assert param.dtype == torch.float64
        else:
            print(f"Unsupported dtype: {PARAMETERS['dtype']}")
            sys.exit(1)



def write_xyz(coords, symbols, file_name="test.xyz"):
    num_atoms = len(symbols)
    assert len(coords) == num_atoms
    with open(file_name, "w") as file:
        file.write(str(num_atoms) + "\n")
        file.write("\n")
        for ida in range(num_atoms):
            file.write(
                symbols[ida]
                + " "
                + str(coords[ida][0])
                + " "
                + str(coords[ida][1])
                + " "
                + str(coords[ida][2])
                + "\n"
            )
    return file_name
