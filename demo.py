import sieve
import argparse
import gc
import logging
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import torch
import torchaudio
from mmaudio.eval_utils import (ModelConfig, VideoInfo, all_model_cfg, generate, load_image,
                                load_video, make_video, setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.sequence_config import SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils


@sieve.function(
    name="mmaudio-app",
    python_packages=["gradio", "torch", "torchaudio", "colorlog", "av", "torchdiffeq", "einops",
                     "open_clip_torch", "omegaconf", "librosa"],
    system_packages=["ffmpeg", "git", "git-lfs"],
    gpu=sieve.gpu.A100(split=8),
    cuda_version="12.1",
    python_version="3.10"
)
def video_to_audio(video_file: sieve.File, prompt: str, negative_prompt: str, seed: int, num_steps: int,
                   cfg_strength: float, duration: float):
    """
    Generates audio for a given video based on a textual prompt.

    :param video_file: Input video file to generate audio for
    :param prompt: Text prompt for audio generation
    :param negative_prompt: Negative text prompt
    :param seed: Seed for random number generation
    :param num_steps: Number of generation steps
    :param cfg_strength: Guidance strength
    :param duration: Duration of the generated audio in seconds
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    log = logging.getLogger()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.bfloat16

    model: ModelConfig = all_model_cfg['large_44k_v2']
    model.download_if_needed()
    output_dir = Path('./output/video_to_audio')
    output_dir.mkdir(exist_ok=True, parents=True)

    setup_eval_logging()

    def get_model() -> tuple[MMAudio, FeaturesUtils, SequenceConfig]:
        seq_cfg = model.seq_cfg

        net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
        net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
        log.info(f'Loaded weights from {model.model_path}')

        feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                      synchformer_ckpt=model.synchformer_ckpt,
                                      enable_conditions=True,
                                      mode=model.mode,
                                      bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                      need_vae_encoder=False)
        feature_utils = feature_utils.to(device, dtype).eval()

        return net, feature_utils, seq_cfg

    net, feature_utils, seq_cfg = get_model()

    @torch.inference_mode()
    def process_video():
        rng = torch.Generator(device=device)
        if seed >= 0:
            rng.manual_seed(seed)
        else:
            rng.seed()
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

        # Use video_file.path to get the local path to the uploaded file
        video_info = load_video(video_file.path, duration)
        clip_frames = video_info.clip_frames
        sync_frames = video_info.sync_frames
        video_duration = video_info.duration_sec
        clip_frames = clip_frames.unsqueeze(0)
        sync_frames = sync_frames.unsqueeze(0)
        seq_cfg.duration = video_duration
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

        audios = generate(clip_frames,
                          sync_frames, [prompt],
                          negative_text=[negative_prompt],
                          feature_utils=feature_utils,
                          net=net,
                          fm=fm,
                          rng=rng,
                          cfg_strength=cfg_strength)
        audio = audios.float().cpu()[0]

        current_time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_save_path = output_dir / f'{current_time_string}.mp4'
        make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
        gc.collect()
        return sieve.File(path=str(video_save_path))  # Return as sieve.File

    return process_video()