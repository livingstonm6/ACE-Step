"""Fast ACE-Step Pipeline with pre-compiled models for optimized loading.

This module provides an optimized version of ACEStepPipeline that loads pre-compiled
models to significantly reduce initialization time in containerized environments.
"""

import os
import torch
from loguru import logger
from .pipeline_ace_step import ACEStepPipeline
from .models.ace_step_transformer import ACEStepTransformer2DModel
from .music_dcae.music_dcae_pipeline import MusicDCAE
from transformers import UMT5EncoderModel, AutoTokenizer
from .language_segmentation import LangSegment, language_filters
from .models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer


class FastACEStepPipeline(ACEStepPipeline):
    """Optimized ACEStepPipeline that loads pre-compiled models for faster initialization."""
    
    def __init__(self, compiled_models_dir, **kwargs):
        """Initialize FastACEStepPipeline with pre-compiled models.
        
        Args:
            compiled_models_dir: Directory containing pre-compiled model artifacts
            **kwargs: Additional arguments passed to parent ACEStepPipeline
        """
        # Initialize parent without loading models
        super().__init__(**kwargs)
        self.compiled_models_dir = compiled_models_dir
        self.loaded = False
        
    def load_compiled_checkpoint(self):
        """Load pre-compiled models for instant startup."""
        logger.info(f"Loading pre-compiled models from: {self.compiled_models_dir}")
        
        # Load pre-compiled ACE Step Transformer
        ace_step_compiled_path = os.path.join(self.compiled_models_dir, "ace_step_transformer_compiled.pt")
        if os.path.exists(ace_step_compiled_path):
            logger.info("Loading pre-compiled ACE Step Transformer")
            self.ace_step_transformer = torch.load(ace_step_compiled_path, map_location=self.device, weights_only=False)
        else:
            logger.warning("Pre-compiled ACE Step Transformer not found, falling back to regular loading")
            self._load_ace_step_transformer_fallback()
            
        # Load pre-compiled Music DCAE
        dcae_compiled_path = os.path.join(self.compiled_models_dir, "music_dcae_compiled.pt")
        if os.path.exists(dcae_compiled_path):
            logger.info("Loading pre-compiled Music DCAE")
            self.music_dcae = torch.load(dcae_compiled_path, map_location=self.device, weights_only=False)
        else:
            logger.warning("Pre-compiled Music DCAE not found, falling back to regular loading")
            self._load_music_dcae_fallback()
            
        # Load pre-compiled Text Encoder
        text_encoder_compiled_path = os.path.join(self.compiled_models_dir, "text_encoder_compiled.pt")
        if os.path.exists(text_encoder_compiled_path):
            logger.info("Loading pre-compiled Text Encoder")
            self.text_encoder_model = torch.load(text_encoder_compiled_path, map_location=self.device, weights_only=False)
        else:
            logger.warning("Pre-compiled Text Encoder not found, falling back to regular loading")
            self._load_text_encoder_fallback()
            
        # Load tokenizer (lightweight, no compilation needed)
        tokenizer_path = os.path.join(self.compiled_models_dir, "tokenizer")
        if os.path.exists(tokenizer_path):
            self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            # Fallback to checkpoint dir
            text_encoder_checkpoint_path = os.path.join(self.checkpoint_dir, "umt5-base")
            self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_checkpoint_path)
            
        # Initialize language segmentation and lyric tokenizer (lightweight)
        lang_segment = LangSegment()
        lang_segment.setfilters(language_filters.default)
        self.lang_segment = lang_segment
        self.lyric_tokenizer = VoiceBpeTokenizer()
        
        self.loaded = True
        logger.info("FastACEStepPipeline loaded successfully")
        
    def _load_ace_step_transformer_fallback(self):
        """Fallback method to load ACE Step Transformer normally."""
        checkpoint_dir = self.get_checkpoint_path(self.checkpoint_dir, "ACE-Step/ACE-Step-v1-3.5B")
        ace_step_checkpoint_path = os.path.join(checkpoint_dir, "ace_step_transformer")
        
        self.ace_step_transformer = ACEStepTransformer2DModel.from_pretrained(
            ace_step_checkpoint_path, torch_dtype=self.dtype
        )
        
        if self.cpu_offload:
            self.ace_step_transformer = self.ace_step_transformer.to("cpu").eval().to(self.dtype)
        else:
            self.ace_step_transformer = self.ace_step_transformer.to(self.device).eval().to(self.dtype)
            
        if self.torch_compile:
            self.ace_step_transformer = torch.compile(self.ace_step_transformer)
            
    def _load_music_dcae_fallback(self):
        """Fallback method to load Music DCAE normally."""
        checkpoint_dir = self.get_checkpoint_path(self.checkpoint_dir, "ACE-Step/ACE-Step-v1-3.5B")
        dcae_checkpoint_path = os.path.join(checkpoint_dir, "music_dcae_f8c8")
        vocoder_checkpoint_path = os.path.join(checkpoint_dir, "music_vocoder")
        
        self.music_dcae = MusicDCAE(
            dcae_checkpoint_path=dcae_checkpoint_path,
            vocoder_checkpoint_path=vocoder_checkpoint_path,
        )
        
        if self.cpu_offload:
            self.music_dcae = self.music_dcae.to("cpu").eval().to(self.dtype)
        else:
            self.music_dcae = self.music_dcae.to(self.device).eval().to(self.dtype)
            
        if self.torch_compile:
            self.music_dcae = torch.compile(self.music_dcae)
            
    def _load_text_encoder_fallback(self):
        """Fallback method to load Text Encoder normally."""
        checkpoint_dir = self.get_checkpoint_path(self.checkpoint_dir, "ACE-Step/ACE-Step-v1-3.5B")
        text_encoder_checkpoint_path = os.path.join(checkpoint_dir, "umt5-base")
        
        text_encoder_model = UMT5EncoderModel.from_pretrained(
            text_encoder_checkpoint_path, torch_dtype=self.dtype
        ).eval()
        
        if self.cpu_offload:
            text_encoder_model = text_encoder_model.to("cpu").eval().to(self.dtype)
        else:
            text_encoder_model = text_encoder_model.to(self.device).eval().to(self.dtype)
            
        text_encoder_model.requires_grad_(False)
        self.text_encoder_model = text_encoder_model
        
        if self.torch_compile:
            self.text_encoder_model = torch.compile(self.text_encoder_model)


def warmup_and_compile_models(
    checkpoint_dir='',
    compiled_models_dir="/tmp/compiled_models",
    device_id="0",
    dtype="bfloat16",
    torch_compile=True,
    **kwargs
):
    """Warmup function to pre-compile models and save them for fast loading.
    
    This function should be called during container build time to pre-compile
    all models and save them as artifacts for instant loading.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        compiled_models_dir: Directory to save compiled model artifacts
        device_id: GPU device ID to use for compilation
        dtype: Model precision ("bfloat16" or "float32")
        torch_compile: Whether to use torch.compile for optimization
        **kwargs: Additional arguments for pipeline initialization
    """
    logger.info("Starting model warmup and compilation process...")
    
    # Ensure output directory exists
    os.makedirs(compiled_models_dir, exist_ok=True)
    
    # Initialize regular pipeline for compilation
    pipeline = ACEStepPipeline(
        checkpoint_dir=checkpoint_dir,
        device_id=device_id,
        dtype=dtype,
        torch_compile=torch_compile,
        **kwargs
    )
    
    # Load and compile models
    logger.info("Loading and compiling models...")
    if pipeline.quantized:
        pipeline.load_quantized_checkpoint(checkpoint_dir)
    else:
        pipeline.load_checkpoint(checkpoint_dir)
    
    # Perform warmup inference to trigger compilation
    logger.info("Performing warmup inference to trigger compilation...")
    try:
        # Create dummy inputs for warmup
        dummy_params = (
            "wav",  # format
            10.0,   # audio_duration
            "ambient",  # prompt
            "[instrumental]",  # lyrics
            20,     # infer_step
            3.0,    # guidance_scale
            "euler",  # scheduler_type
            "apg",  # cfg_type
            1.0,    # omega_scale
            "42",   # actual_seeds
            0.5,    # guidance_interval
            0.0,    # guidance_interval_decay
            3.0,    # min_guidance_scale
            True,   # use_erg_tag
            True,   # use_erg_lyric
            True,   # use_erg_diffusion
            None,  # oss_steps
            0.0,    # guidance_scale_text
            0.0,    # guidance_scale_lyric
        )
        
        # Run a short warmup inference
        warmup_output_path = os.path.join(compiled_models_dir, "warmup_output.wav")
        pipeline(*dummy_params, save_path=warmup_output_path)
        
        # Clean up warmup output
        if os.path.exists(warmup_output_path):
            os.remove(warmup_output_path)
            
    except Exception as e:
        logger.warning(f"Warmup inference failed, but continuing with model saving: {e}")
    
    # Save compiled models
    logger.info("Saving compiled models...")
    
    # Save ACE Step Transformer
    ace_step_compiled_path = os.path.join(compiled_models_dir, "ace_step_transformer_compiled.pt")
    torch.save(pipeline.ace_step_transformer, ace_step_compiled_path)
    logger.info(f"Saved compiled ACE Step Transformer to: {ace_step_compiled_path}")
    
    # Save Music DCAE
    dcae_compiled_path = os.path.join(compiled_models_dir, "music_dcae_compiled.pt")
    torch.save(pipeline.music_dcae, dcae_compiled_path)
    logger.info(f"Saved compiled Music DCAE to: {dcae_compiled_path}")
    
    # Save Text Encoder
    text_encoder_compiled_path = os.path.join(compiled_models_dir, "text_encoder_compiled.pt")
    torch.save(pipeline.text_encoder_model, text_encoder_compiled_path)
    logger.info(f"Saved compiled Text Encoder to: {text_encoder_compiled_path}")
    
    # Save tokenizer
    tokenizer_path = os.path.join(compiled_models_dir, "tokenizer")
    pipeline.text_tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Saved tokenizer to: {tokenizer_path}")
    
    # Clean up memory
    pipeline.cleanup_memory()
    del pipeline
    
    logger.info(f"Model warmup and compilation completed. Artifacts saved to: {compiled_models_dir}")
    return compiled_models_dir


if __name__ == "__main__":
    # Example usage for container build
    import argparse
    
    parser = argparse.ArgumentParser(description="Warmup and compile ACE-Step models")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--compiled-models-dir", type=str, default="/tmp/compiled_models", 
                       help="Directory to save compiled models")
    parser.add_argument("--device-id", type=int, default="0", help="GPU device ID")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--no-torch-compile", action="store_true", help="Disable torch.compile")
    
    args = parser.parse_args()
    
    warmup_and_compile_models(
        checkpoint_dir=args.checkpoint_dir,
        compiled_models_dir=args.compiled_models_dir,
        device_id=args.device_id,
        dtype=args.dtype,
        torch_compile=not args.no_torch_compile
    )