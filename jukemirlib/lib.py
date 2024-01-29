"""Extraction methods here."""

import librosa as lr
import torch
import torch as t
import gc
import numpy as np

from . import constants as c
from .setup_models import *

__all__ = ["load_audio", "extract"]

JUKEBOX_SAMPLE_RATE = 44100
VQVAE_RATE = JUKEBOX_SAMPLE_RATE / 128


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def load_audio(fpath, offset=0.0, duration=None):
    if duration is not None:
        audio, _ = lr.load(
            fpath, sr=JUKEBOX_SAMPLE_RATE, offset=offset, duration=duration
        )
    else:
        audio, _ = lr.load(fpath, sr=JUKEBOX_SAMPLE_RATE, offset=offset)

    if audio.ndim == 1:
        audio = audio[np.newaxis]
    audio = audio.mean(axis=0)

    # normalize audio
    norm_factor = np.abs(audio).max()
    if norm_factor > 0:
        audio /= norm_factor

    return audio.flatten()


def get_z(vqvae, audio, max_length=8192):
    # don't compute unnecessary discrete encodings
    if type(audio) == type([]):
        audio = np.array(audio)[:, :max_length * 128]
    else:
        audio = audio[np.newaxis, :max_length * 128]

    audio = torch.from_numpy(audio[..., np.newaxis]).to(device=c.DEVICE)

    zs = vqvae.encode(audio)

    # get the last level of VQ-VAE tokens
    z = zs[-1]

    return z


def get_cond(top_prior):
    # model only accepts sample length conditioning of
    # >60 seconds, so we use 62
    sample_length_in_seconds = 62

    HPS_SAMPLE_LENGTH = (
        int(sample_length_in_seconds * JUKEBOX_SAMPLE_RATE) // top_prior.raw_to_tokens
    ) * top_prior.raw_to_tokens

    # NOTE: the 'lyrics' parameter is required, which is why it is included,
    # but it doesn't actually change anything about the `x_cond`, `y_cond`,
    # nor the `prime` variables. The `prime` variable is supposed to represent
    # the lyrics, but the LM prior we're using does not condition on lyrics,
    # so it's just an empty tensor.
    metas = [
        dict(
            artist="unknown",
            genre="unknown",
            total_length=HPS_SAMPLE_LENGTH,
            offset=0,
            lyrics="""placeholder lyrics""",
        ),
    ] * 8

    labels = [None, None, top_prior.labeller.get_batch_labels(metas, c.DEVICE)]

    x_cond, y_cond, prime = top_prior.get_cond(None, top_prior.get_y(labels[-1], 0))

    x_cond = x_cond[0, :top_prior.n_ctx][np.newaxis, ...].to(c.DEVICE)
    y_cond = y_cond[0][np.newaxis, ...].to(c.DEVICE)

    return x_cond, y_cond


def downsample(representation, target_rate=30, method=None):
    if method is None:
        method = "librosa_fft"

    if method == "librosa_kaiser":
        resampled_reps = lr.resample(
            np.asfortranarray(representation.T), VQVAE_RATE, target_rate
        ).T
    elif method in ["librosa_fft", "librosa_scipy"]:
        resampled_reps = lr.resample(
            np.asfortranarray(representation.T),
            orig_sr=VQVAE_RATE,
            target_sr=target_rate,
            res_type="fft",
        ).T
    elif method == "mean":
        raise NotImplementedError

    return resampled_reps


def roll(x, n):
    return t.cat((x[:, -n:], x[:, :-n]), dim=1)


def get_activations_custom(
    x, x_cond, y_cond, layers_to_extract=None, fp16=False, fp16_out=False
):
    from .constants import TOP_PRIOR

    # this function is adapted from:
    # https://github.com/openai/jukebox/blob/08efbbc1d4ed1a3cef96e08a931944c8b4d63bb3/jukebox/prior/autoregressive.py#L116

    # custom jukemir stuff
    if layers_to_extract is None:
        layers_to_extract = [36]

    x = x[:, :TOP_PRIOR.n_ctx]  # limit to max context window of Jukebox

    input_seq_length = x.shape[1]

    # chop x_cond if input is short
    x_cond = x_cond[:, :input_seq_length]

    # Preprocess.
    with t.no_grad():
        x = TOP_PRIOR.prior.preprocess(x)

    N, D = x.shape
    # assert isinstance(x, t.cuda.LongTensor)
    assert isinstance(x, t.cuda.LongTensor) or isinstance(x, t.LongTensor)
    assert (0 <= x).all() and (x < TOP_PRIOR.prior.bins).all()

    if TOP_PRIOR.prior.y_cond:
        assert y_cond is not None
        assert y_cond.shape == (N, 1, TOP_PRIOR.prior.width)
    else:
        assert y_cond is None

    if TOP_PRIOR.prior.x_cond:
        assert x_cond is not None
        assert x_cond.shape == (N, D, TOP_PRIOR.prior.width) or x_cond.shape == (
            N,
            1,
            TOP_PRIOR.prior.width,
        ), f"{x_cond.shape} != {(N, D, TOP_PRIOR.prior.width)} nor {(N, 1, TOP_PRIOR.prior.width)}. Did you pass the correct --sample_length?"
    else:
        assert x_cond is None
        x_cond = t.zeros((N, 1, TOP_PRIOR.prior.width), device=x.device, dtype=t.float)

    # self.x_emb is just a straightforward embedding, no trickery here
    x = TOP_PRIOR.prior.x_emb(x)  # X emb
    # this is to be able to fit in a start token/conditioning info: just shift to the right by 1
    x = roll(x, 1)  # Shift by 1, and fill in start token
    # self.y_cond == True always, so we just use y_cond here
    if TOP_PRIOR.prior.y_cond:
        x[:, 0] = y_cond.view(N, TOP_PRIOR.prior.width)
    else:
        x[:, 0] = TOP_PRIOR.prior.start_token

    # for some reason, p=0.0, so the dropout stuff does absolutely nothing
    x = (
        TOP_PRIOR.prior.x_emb_dropout(x)
        + TOP_PRIOR.prior.pos_emb_dropout(TOP_PRIOR.prior.pos_emb())[:input_seq_length]
        + x_cond
    )  # Pos emb and dropout

    layers = TOP_PRIOR.prior.transformer._attn_mods

    reps = {}

    if fp16:
        x = x.half()

    for i, l in enumerate(layers):
        # to be able to take in shorter clips, we set sample to True,
        # but as a consequence the forward function becomes stateful
        # and its state changes when we apply a layer (attention layer
        # stores k/v's to cache), so we need to clear its cache religiously
        l.attn.del_cache()

        x = l(x, encoder_kv=None, sample=True)

        l.attn.del_cache()

        if i + 1 in layers_to_extract:
            reps[i + 1] = np.array(x.squeeze().cpu())

            # break if this is the last one we care about
            if layers_to_extract.index(i + 1) == len(layers_to_extract) - 1:
                break

    return reps


# important, gradient info takes up too much space,
# causes CUDA OOM
@torch.no_grad()
def extract(
    # audio and fpath can be either a single item
    # or a list of items. will attempt batch inference
    # when provided a list.
    audio=None,
    fpath=None,
    meanpool=False,
    # pick which layer(s) to extract from
    layers=None,
    # pick which part of the clip to load in
    offset=0.0,
    duration=None,
    # downsampling frame-wise reps
    downsample_target_rate=None,
    downsample_method=None,
    # for speed-saving
    fp16=False,
    # for space-saving
    fp16_out=False,
    # for GPU VRAM. potentially slows it
    # down but we clean up garbage VRAM.
    # disable if your GPU has a lot of memory
    # or if you're extracting from earlier
    # layers.
    force_empty_cache=True,
):
    # set up the models if they have not been yet
    if c.VQVAE is None and c.TOP_PRIOR is None:
        c.VQVAE, c.TOP_PRIOR = setup_models()

    # main function that runs extraction end-to-end.

    if layers is None:
        layers = [36]  # by default

    assert (audio is None) != (fpath is None), "Must provide either audio or fpath"

    if audio is None:
        if type(fpath) == type([]):
            audio = [
                load_audio(path, offset=offset, duration=duration) for path in fpath
            ]
            bsize = len(fpath)
        else:
            audio = load_audio(fpath, offset=offset, duration=duration)
            bsize = 1

    elif fpath is None:
        if type(audio) == type([]):
            bsize = len(audio)
        else:
            bsize = 1

    if force_empty_cache:
        empty_cache()

    # run vq-vae on the audio to get discretized audio
    z = get_z(c.VQVAE, audio, max_length=c.TOP_PRIOR.n_ctx)

    if force_empty_cache:
        empty_cache()

    # get conditioning info
    x_cond, y_cond = get_cond(c.TOP_PRIOR)

    # avoid raising asserts
    x_cond, y_cond = x_cond.repeat(bsize, 1, 1), y_cond.repeat(bsize, 1, 1)

    if force_empty_cache:
        empty_cache()

    # get the activations from the LM
    acts = get_activations_custom(
        z, x_cond, y_cond, layers_to_extract=layers, fp16=fp16, fp16_out=fp16_out
    )

    if force_empty_cache:
        empty_cache()

    # postprocessing
    if downsample_target_rate is not None:
        for num in acts.keys():
            if bsize == 1:
                acts[num] = downsample(
                    acts[num],
                    target_rate=downsample_target_rate,
                    method=downsample_method,
                )
            else:
                acts[num] = np.array(
                    [
                        downsample(
                            act,
                            target_rate=downsample_target_rate,
                            method=downsample_method,
                        )
                        for act in acts[num]
                    ]
                )

    if meanpool:
        acts = {num: act.mean(axis=0) for num, act in acts.items()}

    if not fp16_out:
        acts = {num: act.astype(np.float32) for num, act in acts.items()}

    return acts
