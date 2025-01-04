import os
import torch
from pyannote.audio import Pipeline
from triton_python_backend_utils import TritonPythonModel, get_input_tensor_by_name, Tensor

class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model."""
        # Directory to save intermediate files
        self.output_dir = os.path.join(args['model_repository'], args['model_version'], 'output')
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize the pyannote.audio pipeline
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="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"
        )
        self.pipeline.to(torch.device("cuda"))

    def execute(self, requests):
        """Execute the model on input requests."""
        responses = []

        for request in requests:
            # Extract input audio tensor
            input_tensor = get_input_tensor_by_name(request, "AUDIO_INPUT")
            audio_data = input_tensor.as_numpy()

            # Save input audio to a temporary file
            audio_file_path = os.path.join(self.output_dir, "input_audio.wav")
            with open(audio_file_path, "wb") as audio_file:
                audio_file.write(audio_data.tobytes())

            # Perform diarization
            diarization = self.pipeline(audio_file_path)

            # Save diarization output to RTTM format
            rttm_file_path = os.path.join(self.output_dir, "output_audio.rttm")
            with open(rttm_file_path, "w") as rttm_file:
                diarization.write_rttm(rttm_file)

            # Read RTTM file as binary and create output tensor
            with open(rttm_file_path, "rb") as rttm_file:
                rttm_data = rttm_file.read()

            output_tensor = Tensor("RTTM_OUTPUT", rttm_data)
            responses.append(output_tensor)

        return responses

    def finalize(self):
        """Clean up resources."""
        # Optionally, clean up temporary files or directories
        if os.path.exists(self.output_dir):
            for file_name in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, file_name)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            os.rmdir(self.output_dir)
