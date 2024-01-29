import os
from tqdm import tqdm
import wget
import sys
import gc

import jukebox
from jukebox.hparams import setup_hparams
from jukebox.make_models import MODELS, make_prior, make_vqvae
import jukebox.utils.dist_utils

from accelerate import init_empty_weights

import torch.nn as nn
import torch

from . import constants


__all__ = ["setup_models"]

# print_once is intended to be used in a distributed context; disable it
jukebox.utils.dist_utils.print_once = print


# this is a huggingface accelerate method, all we do is just
# remove the type hints that we don't want to import in the header
def set_module_tensor_to_device(
    module: nn.Module, tensor_name: str, device, value=None
):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).
    Args:
        module (`torch.nn.Module`): The module in which the tensor we want to move lives.
        param_name (`str`): The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`): The device on which to set the tensor.
        value (`torch.Tensor`, *optional*): The value of the tensor (useful when going from the meta device to any
            other device).
    """
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(
            f"{module} does not have a parameter or a buffer named {tensor_name}."
        )
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if (
        old_value.device == torch.device("meta")
        and device not in ["meta", torch.device("meta")]
        and value is None
    ):
        raise ValueError(
            f"{tensor_name} is on the meta device, we need a `value` to put in on {device}."
        )

    with torch.no_grad():
        if value is None:
            new_value = old_value.to(device)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)

        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif (
            value is not None
            or torch.device(device) != module._parameters[tensor_name].device
        ):
            param_cls = type(module._parameters[tensor_name])
            kwargs = module._parameters[tensor_name].__dict__
            new_value = param_cls(
                new_value, requires_grad=old_value.requires_grad, **kwargs
            ).to(device)
            module._parameters[tensor_name] = new_value


def get_checkpoint(local_path, remote_prefix):
    if not os.path.exists(local_path):
        remote_path = remote_prefix + local_path.split("/")[-1]

        # create this bar_progress method which is invoked automatically from wget
        def bar_progress(current, total, width=80):
            progress_message = "Downloading: %d%% [%d / %d] bytes" % (
                current / total * 100,
                current,
                total,
            )
            # Don't use print() as it will print in new line every time.
            sys.stdout.write("\r" + progress_message)
            sys.stdout.flush()

        wget.download(remote_path, local_path, bar=bar_progress)


def load_weights(model, weights_path, device):
    model_weights = torch.load(weights_path, map_location="cpu")

    # load_state_dict, basically
    for k in tqdm(model_weights["model"].keys()):
        set_module_tensor_to_device(model, k, device, value=model_weights["model"][k])

    model.to(device)

    del model_weights


def setup_models(cache_dir=None, remote_prefix=None, model="5b", device=None, verbose=True):
    if cache_dir is None:
        cache_dir = constants.CACHE_DIR

    if remote_prefix is None:
        remote_prefix = constants.REMOTE_PREFIX

    if device is None:
        device = constants.DEVICE

    # caching preliminaries
    cache_dir = cache_dir + "/" + model
    vqvae_cache_path = cache_dir + "/vqvae.pth.tar"
    prior_cache_path = cache_dir + "/prior_level_2.pth.tar"
    os.makedirs(cache_dir, exist_ok=True)

    # get the checkpoints downloaded if they haven't been already
    get_checkpoint(vqvae_cache_path, remote_prefix + "5b/")
    get_checkpoint(prior_cache_path, remote_prefix + model + "/")

    if verbose:
        print("Importing jukebox and associated packages...")

    priors = MODELS[model]
    prior_hparams = setup_hparams(priors[-1], dict())
    vqvae_hparams = setup_hparams(priors[0], dict(sample_length=prior_hparams.n_ctx * 128))

    # Set up VQVAE
    if verbose:
        print("Setting up the VQ-VAE...")

    # don't actually load any weights in yet,
    # leave it for later. memory optimization
    with init_empty_weights():
        constants.VQVAE = make_vqvae(vqvae_hparams, "meta")

    # Set up language model
    if verbose:
        print("Setting up the top prior...")

    # don't actually load any weights in yet,
    # leave it for later. memory optimization
    with init_empty_weights():
        constants.TOP_PRIOR = make_prior(prior_hparams, constants.VQVAE, "meta")

    # flips a bit that tells the model to return activations
    # instead of projecting to tokens and getting loss for
    # forward pass
    constants.TOP_PRIOR.prior.only_encode = True

    if verbose:
        print("Loading the top prior weights into memory...")

    load_weights(constants.TOP_PRIOR, prior_cache_path, device)

    gc.collect()
    torch.cuda.empty_cache()

    load_weights(constants.VQVAE, vqvae_cache_path, device)
