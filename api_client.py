#!/usr/bin/env python3
"""
ACEStep API Client

This script demonstrates how to send requests to the ACEStep Pipeline API
defined in infer-api.py using the requests library.
"""

import requests
import json
import time
from typing import List, Optional


class ACEStepAPIClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize the API client.
        
        Args:
            base_url: The base URL of the ACEStep API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def health_check(self) -> dict:
        """
        Check if the API server is healthy.
        
        Returns:
            dict: Health status response
        """
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "unhealthy"}
    
    def get_status(self) -> dict:
        """Get detailed API status including pipeline information."""
        try:
            response = self.session.get(f"{self.base_url}/status")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def warmup_models(
        self,
        checkpoint_path: str = None,
        compiled_models_dir: str = None,
        device_id: int = 0,
        bf16: bool = True,
        torch_compile: bool = True
    ) -> dict:
        """Warmup and compile models for faster subsequent loading."""
        try:
            payload = {
                "checkpoint_path": checkpoint_path,
                "compiled_models_dir": compiled_models_dir,
                "device_id": device_id,
                "bf16": bf16,
                "torch_compile": torch_compile
            }
            
            response = self.session.post(
                f"{self.base_url}/warmup",
                json=payload,
                timeout=600  # Warmup can take a while
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def reset_pipeline(self) -> dict:
        """Reset the pipeline instance to force reinitialization."""
        try:
            response = self.session.post(f"{self.base_url}/reset-pipeline")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def generate_audio(
        self,
        checkpoint_path: str,
        prompt: str,
        lyrics: str = "",
        audio_duration: float = 30.0,
        format: str = "wav",
        bf16: bool = True,
        torch_compile: bool = False,
        device_id: int = 0,
        output_path: Optional[str] = None,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: float = 10.0,
        actual_seeds: List[int] = None,
        guidance_interval: float = 0.5,
        guidance_interval_decay: float = 0.0,
        min_guidance_scale: float = 3.0,
        use_erg_tag: bool = True,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        oss_steps: List[int] = None,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        timeout: int = 300
    ) -> dict:
        """
        Generate audio using the ACEStep Pipeline API.
        
        Args:
            checkpoint_path: Path to the model checkpoint directory
            prompt: Text prompt for music generation
            lyrics: Lyrics for the music (optional)
            audio_duration: Duration of generated audio in seconds
            format: Output audio format (wav, mp3, etc.)
            bf16: Use bfloat16 precision
            torch_compile: Enable torch compilation
            device_id: CUDA device ID
            output_path: Custom output path (optional)
            infer_step: Number of inference steps
            guidance_scale: Guidance scale for generation
            scheduler_type: Type of scheduler to use
            cfg_type: CFG type (apg, cfg, etc.)
            omega_scale: Omega scale parameter
            actual_seeds: List of random seeds
            guidance_interval: Guidance interval
            guidance_interval_decay: Guidance interval decay
            min_guidance_scale: Minimum guidance scale
            use_erg_tag: Use ERG tag
            use_erg_lyric: Use ERG lyric
            use_erg_diffusion: Use ERG diffusion
            oss_steps: OSS steps list
            guidance_scale_text: Text guidance scale
            guidance_scale_lyric: Lyric guidance scale
            timeout: Request timeout in seconds
            
        Returns:
            dict: API response containing status and output path
        """
        # Set default values for lists if None
        if actual_seeds is None:
            actual_seeds = [42]
        if oss_steps is None:
            oss_steps = []
            
        # Prepare the request payload
        payload = {
            "format": format,
            "checkpoint_path": checkpoint_path,
            "bf16": bf16,
            "torch_compile": torch_compile,
            "device_id": device_id,
            "output_path": output_path,
            "audio_duration": audio_duration,
            "prompt": prompt,
            "lyrics": lyrics,
            "infer_step": infer_step,
            "guidance_scale": guidance_scale,
            "scheduler_type": scheduler_type,
            "cfg_type": cfg_type,
            "omega_scale": omega_scale,
            "actual_seeds": actual_seeds,
            "guidance_interval": guidance_interval,
            "guidance_interval_decay": guidance_interval_decay,
            "min_guidance_scale": min_guidance_scale,
            "use_erg_tag": use_erg_tag,
            "use_erg_lyric": use_erg_lyric,
            "use_erg_diffusion": use_erg_diffusion,
            "oss_steps": oss_steps,
            "guidance_scale_text": guidance_scale_text,
            "guidance_scale_lyric": guidance_scale_lyric
        }
        
        try:
            print(f"Sending request to {self.base_url}/generate...")
            print(f"Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            return {"error": f"Request timed out after {timeout} seconds"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse response JSON: {str(e)}"}


def main():
    """
    Example usage of the ACEStep API client.
    """
    # Initialize the client
    client = ACEStepAPIClient("http://localhost:8001")
    
    # Check if the server is healthy
    print("Checking server health...")
    health = client.health_check()
    print(f"Health check result: {health}")
    
    if "error" in health:
        print("Server is not healthy. Please start the API server first.")
        print("Run: python infer-api.py")
        return
    
    # Example generation request
    print("\nGenerating audio...")
    
    # Update this path to your actual checkpoint directory
    checkpoint_path = "/path/to/your/checkpoint/directory"

    start_time = time.perf_counter()
    
    result = client.generate_audio(
        checkpoint_path=checkpoint_path,
        prompt="A cheerful pop song with upbeat rhythm",
        lyrics="[instrumental]",
        audio_duration=30.0,
        infer_step=60,  # Reduced for faster testing
        output_path="generated_music.wav"
    )
    gen_time = time.perf_counter() - start_time

    print(f"Generation finished in {gen_time} seconds")
    
    print(f"\nGeneration result: {json.dumps(result, indent=2)}")
    
    if result.get("status") == "success":
        print(f"\nAudio generated successfully!")
        print(f"Output file: {result.get('output_path')}")
    else:
        print(f"\nGeneration failed: {result.get('message', 'Unknown error')}")


if __name__ == "__main__":
    # Example usage
    client = ACEStepAPIClient()
    
    print("=" * 60)
    print("ACEStep API Client Example")
    print("=" * 60)
    
    # Check API health
    print("\n1. Checking API health...")
    health = client.health_check()
    print(f"Health: {health}")
    
    # Get detailed status
    print("\n2. Getting API status...")
    status = client.get_status()
    print(f"Status: {status}")
    
    before_warmup_time = time.perf_counter()
    # Warmup models (optional, for first-time setup)
    # print("\n3. Warming up models (optional)...")
    # warmup_result = client.warmup_models(
    #     checkpoint_path=None,  # Use default
    #     compiled_models_dir="/tmp/compiled_models",
    #     device_id=0,
    #     bf16=True,
    #     torch_compile=True
    # )
    # end_warmup_time = time.perf_counter()
    # print(f"Warmup result: {warmup_result}")
    # print(f'Warmup execution time: {end_warmup_time - before_warmup_time} seconds')
    
    # Example generation request
    before_generation_time = time.perf_counter()
    print("\n4. Generating audio...")
    result = client.generate_audio(
        format="wav",
        checkpoint_path="/path/to/checkpoints",
        audio_duration=60.0,
        prompt="epic orchestral music with powerful drums",
        lyrics="[instrumental]",
        infer_step=120,
        # guidance_scale=3.0,
        # scheduler_type="euler",
        # cfg_type="cfg",
        # omega_scale=1.0,
        # actual_seeds=[42],
        # guidance_interval=0.5,
        # guidance_interval_decay=0.1,
        # min_guidance_scale=1.0,
        # use_erg_tag=True,
        # use_erg_lyric=True,
        # use_erg_diffusion=True,
        # oss_steps=[10, 20]
    )
    end_generation_time = time.perf_counter()
    print(f"\nGeneration Result: {result}")
    print(f"Generation time: {end_generation_time - before_generation_time} seconds")
    
    # Check status after generation
    print("\n5. Final status check...")
    final_status = client.get_status()
    print(f"Final status: {final_status}")
    
    print("\n✅ Example completed!")