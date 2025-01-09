import os
import json
import torch
import tempfile
from pyannote.audio import Pipeline
import triton_python_backend_utils as pb_utils  # noqa


class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model."""
        self.tmp = tempfile.TemporaryDirectory()
        self.temp_folder = self.tmp.name
        self.device = 'cuda'
        
        # Directory to save intermediate files
        self.output_dir = os.path.join(args['model_repository'], args['model_version'], 'output')
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize the pyannote.audio pipeline
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="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"
        )
        self.pipeline.to(torch.device(self.device))
        self.logger = pb_utils.Logger

    def execute(self, requests):
        """Execute the model on input requests."""
        responses = []

        for request in requests:
            # Get audio data
            wav = pb_utils.get_input_tensor_by_name(request, "WAV").as_numpy()
            
            # Save input audio to a temporary file
            audio_file_path = os.path.join(self.temp_folder, "input_audio.wav")
            with open(audio_file_path, "wb") as audio_file:
                audio_file.write(wav.tobytes())

            # Perform diarization
            diarization = self.pipeline(audio_file_path)

            # Save diarization output to RTTM format
            rttm_file_path = os.path.join(self.output_dir, "output_audio.rttm")
            with open(rttm_file_path, "w") as rttm_file:
                diarization.write_rttm(rttm_file)

            # Read RTTM file as binary and create output tensor
            with open(rttm_file_path, "rb") as rttm_file:
                rttm_data = rttm_file.read()

            # Convert result to tensor and create inference response
            out = [pb_utils.Tensor("RTTM_OUTPUT", rttm_data)]
            inference_response = pb_utils.InferenceResponse(
                output_tensors=out
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """Clean up resources."""
        # Optionally, clean up temporary files or directories
        if os.path.exists(self.temp_folder):
            for file_name in os.listdir(self.temp_folder):
                file_path = os.path.join(self.temp_folder, file_name)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            os.rmdir(self.temp_folder)
