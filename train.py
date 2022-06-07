from argparse import ArgumentParser
from functools import partial
from importlib import import_module
from itertools import product
from os import sep
from pathlib import Path
from typing import Dict, Mapping, Union, Optional, TypeVar

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from data import DatasetSceneMaker
from trajectron_plus_plus import TrajectronPlusPlus, Mode
from trajectron_plus_plus import NodeTypeIterableDataset
from trajectron_plus_plus.data import NodeTypeEnum
from trajectron_plus_plus.data import get_attention_radius, \
    get_standardization_params
from trajectron_plus_plus.data import collate
from trajectron_plus_plus.training import scheduling
from trajectron_plus_plus.training import parameters_but, parameters_of
from trajectron_plus_plus.training import elbo_loss, nll_loss
from trajectron_plus_plus.training import metrics


T = TypeVar("T")

STAGES = ("train", "val", "test")


def train(
        config: Mapping,
        data_path: Union[str, Path],
        logging_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        verbose: bool = False
) -> Dict[str, Dict[str, float]]:
    """

    Args:
        config:
        data_path:
        logging_path:
        checkpoint_path
        verbose:

    Returns:

    """
    # Display the configuration
    if verbose:
        print("Configuration", "\n")
        print(yaml.dump(config), "\n")
    
    # Import the right dataset processing module
    camel_dataset_name = "".join(
        [term.title() for term in config["dataset"]["name"].split("_")]
    )
    dataset_module = import_module(f"data.{config['dataset']['name']}")
    dataset_scene_maker: DatasetSceneMaker = getattr(
        dataset_module,
        camel_dataset_name + "SceneMaker"
    )

    # Compute all edge types
    node_type_enum = NodeTypeEnum(config["dataset"]["agent_types"])
    edge_types = list(product(node_type_enum, repeat=2))

    # Arrange standardization parameters
    standardization_params = get_standardization_params(
        node_type_enum,
        config["model"]["parameters"]
    )

    # Arrange attention radii
    attention_radius = get_attention_radius(
        node_type_enum,
        config["model"]["parameters"]["attention_radius"]
    )

    # Create data loaders
    full_data_path = data_path / Path(config["dataset"]["name"] + sep + "data")
    data_loaders = {
        stage: {
            agent_type: DataLoader(
                NodeTypeIterableDataset(
                    dataset_scene_maker.looping_scene_iterator(
                        str(full_data_path),
                        config["data_loading"][stage]["data"],
                        config["dataset"],
                        node_type_enum,
                        config["model"]["parameters"]["use_maps"],
                        config["data_loading"][stage]["subset"].get("start"),
                        config["data_loading"][stage]["subset"].get("stop"),
                        config["data_loading"][stage]["subset"].get("step")
                    ),
                    agent_type,
                    edge_types,
                    config["model"]["parameters"]["include_robot"],
                    config["model"]["parameters"]["n_history_timesteps_min"],
                    config["model"]["parameters"]["n_history_timesteps_max"],
                    config["model"]["parameters"]["n_future_timesteps_min"],
                    config["model"]["parameters"]["n_future_timesteps_max"],
                    standardization_params,
                    attention_radius,
                    config["model"]["parameters"]["state"],
                    config["model"]["parameters"]["pred_state"],
                    config["model"],
                    config["model"]["parameters"]["node_frequency_multiplier"],
                    config["model"]["parameters"]["scene_frequency_multiplier"],
                    config["model"]["parameters"]["edge_addition_filter"],
                    config["model"]["parameters"]["edge_removal_filter"]
                ),
                batch_size=config["data_loading"][stage]["batch_size"],
                num_workers=config["data_loading"][stage]["n_workers"],
                collate_fn=collate,
                pin_memory=config["training"]["device"] != "cpu",
                drop_last=True
            )
            for agent_type in config["dataset"]["agent_types"]
        } for stage in STAGES
    }

    # Instantiate the model
    tpp = TrajectronPlusPlus(
        config["dataset"]["agent_types"],
        config["dataset"]["ego_agent_type"],
        config["model"]["parameters"]["state"],
        config["model"]["parameters"]["pred_state"],
        config["model"]["parameters"]["include_robot"],
        config["model"]["parameters"]["use_edges"],
        config["model"]["parameters"]["use_maps"],
        config["model"]["architecture"],
    ).to(config["training"]["device"])

    # Create variable schedulers
    schedulers = {}
    for agent_type in config["dataset"]["agent_types"]:
        schedulers[agent_type] = scheduling.SchedulerDict()
        for var, scheduler_params in config["training"]["schedulers"].items():
            # TODO: Check if `pop` works correctly, and does not affect the
            #       configuration
            schedulers[agent_type][var] = getattr(
                scheduling,
                scheduler_params["type"]
            )(**scheduler_params["kwargs"])
    
    # Create the optimizers
    optimizers = {}
    for agent_type, args in config["training"]["optimizers"].items():
        # Segregate parameters
        params = []
        for param in args["kwargs"]["params"]:
            filter_params = parameters_of \
                if param["params"]["type"] == "include" \
                else parameters_but
            param_dict = {
                "params": filter_params(
                    tpp,
                    param["params"]["module"]
                )        
            }
            lr = param.get("lr")
            if lr is not None:
                param_dict["lr"] = lr
            params.append(param_dict)
        
        # Instantiate the agent type optimizer
        optimizers[agent_type] = getattr(optim, args["type"])(
            params=params,
            lr=args["kwargs"]["lr"]
        )
        
    # Create the LR schedulers
    lr_schedulers = {
        agent_type: getattr(
            optim.lr_scheduler,
            config["training"]["lr_schedulers"][agent_type]["type"]
        )(
            optimizers[agent_type],
            **config["training"]["lr_schedulers"][agent_type]["kwargs"]
        ) for agent_type in config["dataset"]["agent_types"]
    }
    
    # Check if there will be gradient clipping
    clip_gradients = config["training"].get("grad_clip_val") is not None

    # Instantiate metrics
    metric_functions = {
        metric_name: partial(
            getattr(metrics, metric_params["type"]),
            **metric_params["kwargs"]
        )
        for metric_name, metric_params
        in config["training"]["metrics"].items()
    }

    # Create the Tensorboard summary writer
    writer = SummaryWriter(logging_path) if logging_path is not None else None

    # Make the checkpoint path a pathlib path, if any
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)
    
    # Initialize the step counters
    i_step = {
        stage: {
            agent_type: 0 for agent_type in config["dataset"]["agent_types"]
        } for stage in STAGES
    }
    n_steps_per_epoch = {
        stage: {
            agent_type: None for agent_type in config["dataset"]["agent_types"]
        } for stage in STAGES
    }
    
    epochs = range(1, config["training"]["n_epochs"] + 1)
    if verbose:
        epochs = tqdm(epochs, desc="Epochs", unit="epoch", position=0)

    def move_to_device(elems: list, device: Union[torch.device, str]):
        for i, elem in enumerate(elems):
            if isinstance(elem, torch.Tensor):
                batch[i] = elem.to(device)
            elif isinstance(elem, dict):
                for edge, sub_elems in elem.items():
                    for j, sub_elem in enumerate(sub_elems):
                        if isinstance(sub_elem, torch.Tensor):
                            batch[i][edge][j] = sub_elem.to(device)
                        elif isinstance(sub_elem, list):
                            batch[i][edge][j] = [
                                t.to(device) for t in sub_elem
                            ]
    
    for e in epochs:

        # Log the beginning of the epoch, if asked
        if writer is not None:
            writer.add_scalar(
                f"epoch",
                e,
                sum(sum(s_step.values()) for s_step in i_step.values())
            )
    
        for stage in STAGES[:-1]:
            tpp.train(stage == "train")

            batch_size = config["data_loading"][stage]["batch_size"]
            
            for agent_type in config["dataset"]["agent_types"]:
                batches = data_loaders[stage][agent_type]
                if verbose:
                    batches = tqdm(
                        batches,
                        desc=" - ".join((stage.title(), agent_type)),
                        total=n_steps_per_epoch[stage][agent_type],
                        leave=False,
                        position=1
                    )

                # Initialize metrics
                n_running_elems = 0
                running_loss = 0.
                metric_values = {
                    metric_name: 0 for metric_name in metric_functions
                }
                
                for batch in batches:
                    # Move the tensors to the right device
                    move_to_device(batch, config["training"]["device"])

                    # Extract the ground truth trajectories
                    ground_truth_trajectories = batch[3]

                    # Check if logging needs to be done at this step
                    log_this_step = (i_step[stage][agent_type] + 1) % (
                        config["training"]["logging"]["log_every_n_steps"][stage]
                    ) == 0 and writer is not None
                    
                    if stage == "train":
                        # Zero gradients
                        optimizers[agent_type].zero_grad()

                        # Update the latent variable clipping value
                        tpp.node_models[agent_type].latent.z_logit_clip = \
                            schedulers[agent_type]["z_logit_clip"].value

                        # Make predictions
                        y_dist, predicted_trajectories = tpp(
                            *batch,
                            config["model"]["parameters"]["n_future_timesteps_max"],
                            n_samples=1,
                            gmm_mode=config["model"]["parameters"]["gmm_mode"]
                        )

                        # Compute the ELBO loss
                        loss = elbo_loss(
                            y_dist,
                            tpp.node_models[agent_type].latent.p,
                            ground_truth_trajectories,
                            tpp.node_models[agent_type].latent.kl_divergence,
                            schedulers[agent_type]["kl_weight"].value,
                            config["training"]["loss"]["max_log_p_yt_xz"]
                        )
                        loss.backward()
                        
                        # Clip gradients, if asked
                        if clip_gradients:
                            nn.utils.clip_grad_value_(
                                tpp.parameters(),
                                config["training"]["grad_clip_val"]
                            )
                        
                        # Step variable schedulers
                        schedulers[agent_type].step()
                        
                        # Step the optimizer
                        optimizers[agent_type].step()

                        # Log the learning rate
                        if log_this_step:
                            lrs = lr_schedulers[agent_type].get_last_lr()
                            for i_lr, lr in enumerate(lrs):
                                writer.add_scalar(
                                    f"lr/{agent_type}_{i_lr}",
                                    lr,
                                    i_step["train"][agent_type]
                                )
                        
                        # Step the LR scheduler
                        lr_schedulers[agent_type].step()
                        
                    else:
                        with torch.no_grad():
                            # Make predictions
                            y_dist, predicted_trajectories = tpp(
                                *batch,
                                config["model"]["parameters"]["n_future_timesteps_max"],
                                n_samples=1,
                                mode=Mode.FULL,
                                gmm_mode=config["model"]["parameters"]["gmm_mode"]
                            )

                            # Compute the NLL loss
                            loss = nll_loss(
                                y_dist,
                                ground_truth_trajectories,
                                config["training"]["loss"]["max_log_p_yt_xz"]
                            )
                            
                    n_running_elems += batch_size
                    running_loss += loss.item()
                            
                    # Update metrics
                    for metric_name, metric in metric_functions.items():
                        metric_values[metric_name] += metric(
                            predicted_trajectories,
                            ground_truth_trajectories,
                        ).sum()
                    
                    # Write the results to the writer
                    if log_this_step:
                        # Log the loss
                        writer.add_scalar(
                            f"{stage}_loss/{agent_type}",
                            running_loss * batch_size / n_running_elems,
                            i_step[stage][agent_type]
                        )

                        # Log metrics
                        for metric_name, metric_value in metric_values.items():
                            writer.add_scalar(
                                f"{stage}_{metric_name}/{agent_type}",
                                metric_value / n_running_elems,
                                i_step[stage][agent_type]
                            )

                        # Reset metric values
                        n_running_elems = 0
                        running_loss = 0
                        metric_values = {
                            metric_name: 0 for metric_name in metric_functions
                        }
                    
                    # Increment the step
                    i_step[stage][agent_type] += 1

                # Update the batch number
                if verbose and n_steps_per_epoch[stage][agent_type] is None:
                    n_steps_per_epoch[stage][agent_type] = \
                        i_step[stage][agent_type]

        # Log the end of the epoch, if asked
        if writer is not None:
            writer.add_scalar(
                f"epoch",
                e,
                sum(sum(s_step.values()) for s_step in i_step.values()) - 1
            )
                    
        # Save a model checkpoint
        if checkpoint_path is not None:
            torch.save(tpp, checkpoint_path / Path(f"epoch_{e}.pt"))
    
    if verbose:
        print("Testing")

    # Create a register to save results for all agent types
    results = {
        agent_type: {
            metric_name: None for metric_name in metric_functions
        } for agent_type in config["dataset"]["agent_types"]
    }

    # Set the model to evaluation mode
    tpp.eval()

    batch_size = config["data_loading"]["test"]["batch_size"]
    
    for agent_type in config["dataset"]["agent_types"]:
        batches = data_loaders["test"][agent_type]
        if verbose:
            batches = tqdm(
                batches,
                desc=" - ".join(("Test", agent_type)),
                leave=False,
            )

        # Reset metric values
        n_running_elems = 0
        metric_values = {
            metric_name: 0 for metric_name in metric_functions
        }

        for batch in batches:
            # Move the tensors to the right device
            move_to_device(batch, config["training"]["device"])

            # Extract the ground truth trajectories
            ground_truth_trajectories = batch[3]

            with torch.no_grad():
                # Make predictions
                y_dist, predicted_trajectories = tpp(
                    *batch,
                    config["model"]["parameters"]["n_future_timesteps_max"],
                    n_samples=config["model"]["parameters"]["n_samples"],
                    mode=Mode.FULL,
                    gmm_mode=config["model"]["parameters"]["gmm_mode"]
                )

                n_running_elems += batch_size

                # Update metrics
                for metric_name, metric in metric_functions.items():
                    metric_values[metric_name] += metric(
                        predicted_trajectories,
                        ground_truth_trajectories,
                    ).sum()

        # Register results for the current agent type
        if n_running_elems > 0:
            for metric_name, metric_value in metric_values.items():
                results[agent_type][metric_name] = \
                    (metric_value / n_running_elems).item()

    return results


if __name__ == "__main__":
    # Create an arguent parser
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment_path",
        help="The path of an experiment directory with a `config.yaml` file",
        type=str
    )
    parser.add_argument(
        "-d",
        "--data_path",
        help="The path pointing out the directory holding datasets",
        type=str,
        default="./data"
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help="Disable verbose",
        action="store_true",
        default=False
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load the experiment configuration
    experiment_path = Path(args.experiment_path).resolve()
    with open(experiment_path / Path("config.yaml"), "r") as f:
        experiment_config = yaml.safe_load(f)

    # Resolve data paths
    data_path = Path(args.data_path).resolve()
    checkpoint_path = (experiment_path / Path("checkpoints")).mkdir(
        parents=True,
        exist_ok=True
    )
    
    # Run the training
    experiment_results = train(
        experiment_config,
        data_path,
        experiment_path,
        checkpoint_path,
        not args.quiet
    )
    
    # Print results
    # TODO: Use tabulate
    print("Results", "\n")
    print(experiment_results)

    # Save results
    with open(experiment_path / Path("results.yaml"), "w") as f:
        yaml.dump(experiment_results, f, default_flow_style=False)
