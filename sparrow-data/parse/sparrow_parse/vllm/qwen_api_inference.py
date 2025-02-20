import os
import json
import requests
from sparrow_parse.vllm.inference_base import ModelInference
from PIL import Image
import base64
from rich import print


class QwenAPIInference(ModelInference):
    """
    A class for performing inference using the Qwen API.
    Handles image preprocessing, response formatting, and API interaction.
    """

    def __init__(self, api_url, api_key):
        """
        Initialize the inference class with API credentials.

        :param api_url: The Qwen API endpoint URL.
        :param api_key: The API key for authentication.
        """
        self.api_url = api_url
        self.api_key = api_key
        print(f"QwenAPIInference initialized with API URL: {api_url}")

    def image_to_data_url(self, image_path):
        """
        Convert an image file to a base64-encoded data URL.

        :param image_path: Path to the image file.
        :return: Base64 encoded data URL string.
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}"

    def process_response(self, output_text):
        """
        Process and clean the API's raw output to format as JSON.

        :param output_text: Raw output text from the API.
        :return: A formatted JSON string or the original text in case of errors.
        """
        try:
            # First try to find JSON content within markdown code blocks
            if "```json" in output_text:
                json_start = output_text.find("```json") + 7
                json_end = output_text.find("```", json_start)
                if json_end != -1:
                    output_text = output_text[json_start:json_end].strip()

            # Clean and parse the text
            cleaned_text = (
                output_text.strip("[]'")
                .replace("'", '"')  # Replace single quotes with double quotes for JSON
            )
            formatted_json = json.loads(cleaned_text)
            return json.dumps(formatted_json, indent=2)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON in Qwen API inference backend: {e}")
            return output_text

    def load_image_data(self, image_filepath, max_width=1250, max_height=1750):
        """
        Load and resize image while maintaining its aspect ratio.

        :param image_filepath: Path to the image file.
        :param max_width: Maximum allowed width of the image.
        :param max_height: Maximum allowed height of the image.
        :return: Tuple containing the image object and its new dimensions.
        """
        image = Image.open(image_filepath)
        width, height = image.size

        # Calculate new dimensions while maintaining the aspect ratio
        if width > max_width or height > max_height:
            aspect_ratio = width / height
            new_width = min(max_width, int(max_height * aspect_ratio))
            new_height = min(max_height, int(max_width / aspect_ratio))
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return image, new_width, new_height

        return image, width, height

    def inference(self, input_data, mode=None):
        """
        Perform inference using the Qwen API.

        :param input_data: A list of dictionaries containing image file paths and text inputs.
        :param mode: Optional mode for inference ("static" for simple JSON output).
        :return: List of processed API responses.
        """
        if mode == "static":
            return [self.get_simple_json()]

        # Prepare absolute file paths
        file_paths = self._extract_file_paths(input_data)
        results = []

        for file_path in file_paths:
            # Load and process image
            image, width, height = self.load_image_data(file_path)
            
            # Convert the image to a data URL
            image_data_url = self.image_to_data_url(file_path)

            # Prepare messages for the API
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at extracting structured text from image documents. Please provide the extracted information in JSON format."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_data[0]["text_input"]},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ]

            # Prepare API request payload
            payload = {
                "model": "qwen/qwen2.5-vl-72b-instruct:free",
                "messages": messages,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            try:
                # Send request to Qwen API
                response = requests.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()

                # Process API response
                response_json = response.json()
                output_text = response_json["choices"][0]["message"]["content"]
                processed_response = self.process_response(output_text)
                results.append(processed_response)

                print("Inference completed successfully for: ", file_path)

            except requests.exceptions.RequestException as e:
                error_msg = f"Qwen API error for {file_path}: {str(e)}"
                print(error_msg)
                results.append({"error": error_msg})

        return results

    @staticmethod
    def _extract_file_paths(input_data):
        """
        Extract and resolve absolute file paths from input data.

        :param input_data: List of dictionaries containing image file paths.
        :return: List of absolute file paths.
        """
        return [
            os.path.abspath(file_path)
            for data in input_data
            for file_path in data.get("file_path", [])
        ]