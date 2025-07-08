from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.fast_pipeline_ace_step import FastACEStepPipeline, warmup_and_compile_models
from acestep.data_sampler import DataSampler
import uuid
from loguru import logger

app = FastAPI(title="ACEStep Pipeline API")

# Global pipeline instance for reuse
_pipeline_instance = None
_compiled_models_dir = os.environ.get("COMPILED_MODELS_DIR", "/tmp/compiled_models")
_use_fast_pipeline = os.environ.get("USE_FAST_PIPELINE", "true").lower() == "true"

class ACEStepInput(BaseModel):
    format: str
    checkpoint_path: str
    bf16: bool = True
    torch_compile: bool = False
    device_id: int = 0
    output_path: Optional[str] = None
    audio_duration: float
    prompt: str
    lyrics: str
    infer_step: int
    guidance_scale: float
    scheduler_type: str
    cfg_type: str
    omega_scale: float
    actual_seeds: List[int]
    guidance_interval: float
    guidance_interval_decay: float
    min_guidance_scale: float
    use_erg_tag: bool
    use_erg_lyric: bool
    use_erg_diffusion: bool
    oss_steps: List[int]
    guidance_scale_text: float = 0.0
    guidance_scale_lyric: float = 0.0

class ACEStepOutput(BaseModel):
    status: str
    output_path: Optional[str]
    message: str

def initialize_pipeline(checkpoint_path: str, bf16: bool, torch_compile: bool, device_id: int) -> ACEStepPipeline:
    """Initialize pipeline with optional fast loading."""
    global _pipeline_instance, _compiled_models_dir, _use_fast_pipeline
    
    # Return existing instance if available
    if _pipeline_instance is not None:
        return _pipeline_instance
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    if _use_fast_pipeline and os.path.exists(_compiled_models_dir):
        logger.info(f"Using FastACEStepPipeline with compiled models from: {_compiled_models_dir}")
        pipeline = FastACEStepPipeline(
            compiled_models_dir=_compiled_models_dir,
            checkpoint_dir=checkpoint_path,
            dtype="bfloat16" if bf16 else "float32",
            torch_compile=torch_compile,
            device_id=device_id
        )
        pipeline.load_compiled_checkpoint()
    else:
        logger.info("Using standard ACEStepPipeline")
        pipeline = ACEStepPipeline(
            checkpoint_dir=checkpoint_path,
            dtype="bfloat16" if bf16 else "float32",
            torch_compile=torch_compile,
            device_id=device_id
        )
        # Load checkpoint if not using fast pipeline
        if not pipeline.loaded:
            if pipeline.quantized:
                pipeline.load_quantized_checkpoint(checkpoint_path)
            else:
                pipeline.load_checkpoint(checkpoint_path)
    
    _pipeline_instance = pipeline
    return pipeline

class WarmupRequest(BaseModel):
    checkpoint_path: Optional[str] = None
    compiled_models_dir: Optional[str] = None
    device_id: str = "0"
    bf16: bool = True
    torch_compile: bool = True


@app.post("/warmup")
async def warmup_models(request: WarmupRequest):
    """Warmup endpoint to pre-compile models for faster subsequent loading."""
    try:
        global _compiled_models_dir
        
        # Use provided compiled_models_dir or default
        target_dir = request.compiled_models_dir if request.compiled_models_dir is not None else _compiled_models_dir
        
        logger.info(f"Starting model warmup process...")
        result_dir = warmup_and_compile_models(
            checkpoint_dir=request.checkpoint_path,
            compiled_models_dir=target_dir,
            device_id=request.device_id,
            dtype="bfloat16" if request.bf16 else "float32",
            torch_compile=request.torch_compile
        )
        
        # Update global compiled models directory
        _compiled_models_dir = result_dir
        
        return {
            "status": "success",
            "message": f"Models warmed up and compiled successfully",
            "compiled_models_dir": result_dir
        }
        
    except Exception as e:
        logger.error(f"Warmup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during warmup: {str(e)}")


@app.post("/generate", response_model=ACEStepOutput)
async def generate_audio(input_data: ACEStepInput):
    try:
        # Initialize or reuse pipeline
        model_demo = initialize_pipeline(
            input_data.checkpoint_path,
            input_data.bf16,
            input_data.torch_compile,
            input_data.device_id
        )

        # Prepare parameters
        params = (
            input_data.format,
            input_data.audio_duration,
            input_data.prompt,
            input_data.lyrics,
            input_data.infer_step,
            input_data.guidance_scale,
            input_data.scheduler_type,
            input_data.cfg_type,
            input_data.omega_scale,
            ", ".join(map(str, input_data.actual_seeds)),
            input_data.guidance_interval,
            input_data.guidance_interval_decay,
            input_data.min_guidance_scale,
            input_data.use_erg_tag,
            input_data.use_erg_lyric,
            input_data.use_erg_diffusion,
            ", ".join(map(str, input_data.oss_steps)),
            input_data.guidance_scale_text,
            input_data.guidance_scale_lyric,
        )

        # Generate output path if not provided
        output_path = input_data.output_path or f"output_{uuid.uuid4().hex}.wav"

        # Run pipeline
        model_demo(
            *params,
            save_path=output_path
        )

        return ACEStepOutput(
            status="success",
            output_path=output_path,
            message="Audio generated successfully"
        )

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

@app.post("/reset-pipeline")
async def reset_pipeline():
    """Reset the global pipeline instance to force reinitialization."""
    global _pipeline_instance
    try:
        if _pipeline_instance is not None:
            # Clean up memory
            _pipeline_instance.cleanup_memory()
            del _pipeline_instance
            _pipeline_instance = None
            logger.info("Pipeline instance reset successfully")
        
        return {
            "status": "success",
            "message": "Pipeline reset successfully"
        }
    except Exception as e:
        logger.error(f"Pipeline reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting pipeline: {str(e)}")


@app.get("/status")
async def get_status():
    """Get current API status and pipeline information."""
    global _pipeline_instance, _compiled_models_dir, _use_fast_pipeline
    
    return {
        "status": "healthy",
        "pipeline_loaded": _pipeline_instance is not None,
        "pipeline_type": "FastACEStepPipeline" if _use_fast_pipeline else "ACEStepPipeline",
        "compiled_models_dir": _compiled_models_dir,
        "compiled_models_available": os.path.exists(_compiled_models_dir) if _compiled_models_dir else False,
        "use_fast_pipeline": _use_fast_pipeline
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
