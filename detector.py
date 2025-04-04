import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor, AutoConfig
from PIL import Image
import requests # For potentially loading images from URLs later, though not implemented in predict yet
from typing import List, Union, Dict, Any
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeepFakeDetector:
    """
    A class to detect deepfakes in images using a pre-trained Hugging Face model.

    Attributes:
        processor: The image processor associated with the model.
        model: The image classification model.
        device: The device (CPU or CUDA) the model is running on.
        id2label: A dictionary mapping class indices to labels (e.g., {0: 'Real', 1: 'Deepfake'}).
    """
    def __init__(self, model_name: str = "prithivMLmods/Deep-Fake-Detector-v2-Model", device: str = None):
        """
        Initializes the DeepFakeDetector.

        Args:
            model_name (str): The name or path of the Hugging Face model to load.
            device (str, optional): The device to run the model on ('cuda', 'cpu'). 
                                    If None, automatically detects CUDA availability.
        """
        logging.info(f"Initializing DeepFakeDetector with model: {model_name}")
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode

            # Load label mapping from model config
            config = AutoConfig.from_pretrained(model_name)
            self.id2label = config.id2label if hasattr(config, 'id2label') else {0: 'Real', 1: 'Deepfake'}
            logging.info(f"Loaded label mapping: {self.id2label}")
            
            logging.info("Model and processor loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model or processor '{model_name}': {e}")
            raise  # Re-raise the exception after logging

    def _load_image(self, image_source: Union[str, Image.Image]) -> Image.Image | None:
        """Loads an image from a file path or returns it if already a PIL Image."""
        try:
            if isinstance(image_source, str):
                if not os.path.exists(image_source):
                    logging.warning(f"Image file not found: {image_source}")
                    return None
                img = Image.open(image_source)
            elif isinstance(image_source, Image.Image):
                img = image_source
            else:
                logging.warning(f"Invalid image source type: {type(image_source)}")
                return None
            
            # Ensure image is in RGB format
            if img.mode != "RGB":
                 img = img.convert("RGB")
            return img
        except FileNotFoundError:
            logging.warning(f"Image file not found during loading: {image_source}")
            return None
        except Image.UnidentifiedImageError:
            logging.warning(f"Cannot identify image file (may be corrupt or not an image): {image_source}")
            return None
        except Exception as e:
            logging.warning(f"Error loading image {image_source}: {e}")
            return None

    def predict(self, image_inputs: Union[Image.Image, str, List[Union[Image.Image, str]]]) -> List[Dict[str, Any]]:
        """
        Predicts whether the input image(s) are real or deepfakes.

        Args:
            image_inputs: A single PIL Image, a single file path, 
                          a list of PIL Images, or a list of file paths.

        Returns:
            A list of dictionaries. Each dictionary contains:
            - 'input_index': The original index of the input image in the batch.
            - 'status': 'success' or 'error'.
            - 'prediction': (if status='success') A dict containing:
                - 'label': The predicted label ('Real' or 'Deepfake').
                - 'confidence': The confidence score (probability) for the predicted label.
                - 'class_id': The predicted class index (e.g., 0 or 1).
            - 'error_message': (if status='error') A description of the error.
            - 'all_probabilities': (if status='success') A dict mapping label names to their probabilities.
        """
        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs] # Make it a list for uniform processing

        results = []
        valid_images = []
        original_indices = [] # Keep track of original position for error reporting

        # 1. Load and validate images
        for i, img_input in enumerate(image_inputs):
            img = self._load_image(img_input)
            if img:
                valid_images.append(img)
                original_indices.append(i)
            else:
                # Add error result directly for invalid inputs
                 results.append({
                    'input_index': i,
                    'status': 'error',
                    'error_message': f"Failed to load or invalid image input at index {i}",
                    'prediction': None,
                    'all_probabilities': None
                 })

        if not valid_images:
            logging.warning("No valid images found in the input batch.")
            # Sort results by original index before returning if all failed
            results.sort(key=lambda x: x['input_index'])
            return results

        # 2. Process valid images in batch
        try:
            inputs = self.processor(images=valid_images, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()} # Move tensors to the correct device
        except Exception as e:
            logging.error(f"Error during image processing: {e}")
            # Add error status for all processed images if processor fails
            for i, original_idx in enumerate(original_indices):
                 results.append({
                    'input_index': original_idx,
                    'status': 'error',
                    'error_message': f"Image processing failed: {e}",
                    'prediction': None,
                    'all_probabilities': None
                 })
            results.sort(key=lambda x: x['input_index'])
            return results
            
        # 3. Perform inference
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 4. Calculate probabilities and predictions
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_indices = probabilities.argmax(-1)
            confidences = probabilities.max(-1).values

            # Move results back to CPU for easier handling
            probabilities_cpu = probabilities.cpu().numpy()
            predicted_indices_cpu = predicted_indices.cpu().numpy()
            confidences_cpu = confidences.cpu().numpy()

            # 5. Format results for valid images
            for i in range(len(valid_images)):
                original_idx = original_indices[i]
                pred_idx = predicted_indices_cpu[i]
                label = self.id2label.get(pred_idx, f"Unknown Class {pred_idx}")
                confidence = confidences_cpu[i]
                all_probs = {self.id2label.get(j, f"Unknown Class {j}"): prob for j, prob in enumerate(probabilities_cpu[i])}

                results.append({
                    'input_index': original_idx,
                    'status': 'success',
                    'prediction': {
                        'label': label,
                        'confidence': float(confidence), # Convert numpy float to python float
                        'class_id': int(pred_idx)
                    },
                     'all_probabilities': {k: float(v) for k,v in all_probs.items()},
                    'error_message': None
                })

        except Exception as e:
            logging.error(f"Error during model inference or result processing: {e}")
            # Add error status for all processed images if inference fails
            for i, original_idx in enumerate(original_indices):
                 # Avoid adding duplicate errors if already added during loading
                 if not any(r['input_index'] == original_idx and r['status'] == 'error' for r in results):
                     results.append({
                        'input_index': original_idx,
                        'status': 'error',
                        'error_message': f"Model inference failed: {e}",
                        'prediction': None,
                        'all_probabilities': None
                     })

        # Sort final results by original input index
        results.sort(key=lambda x: x['input_index'])
        return results

# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy images for testing (replace with your actual image paths)
    try:
        real_image = Image.new('RGB', (224, 224), color = 'red')
        fake_image = Image.new('RGB', (224, 224), color = 'blue')
        real_image.save("dummy_real.png")
        fake_image.save("dummy_fake.png")
        # Create a non-image file
        with open("not_an_image.txt", "w") as f:
            f.write("hello")
    except Exception as e:
        logging.warning(f"Could not create dummy image files (PIL might not be fully installed or permissions issue): {e}")
        # If dummy creation fails, just use None for paths later
        real_image_path = None
        fake_image_path = None
        invalid_path = "non_existent_file.jpg"
        non_image_path = None
    else:
        real_image_path = "dummy_real.png"
        fake_image_path = "dummy_fake.png"
        invalid_path = "non_existent_file.jpg"
        non_image_path = "not_an_image.txt"


    # Initialize the detector
    try:
        detector = DeepFakeDetector() # Automatically uses GPU if available

        # --- Test Cases ---

        # 1. Predict single PIL image
        if real_image:
             print("\n--- Predicting single PIL image ---")
             single_pil_result = detector.predict(real_image)
             print(single_pil_result)

        # 2. Predict single file path
        if real_image_path:
            print("\n--- Predicting single file path ---")
            single_path_result = detector.predict(real_image_path)
            print(single_path_result)

        # 3. Predict batch of PIL images
        if real_image and fake_image:
            print("\n--- Predicting batch of PIL images ---")
            batch_pil_result = detector.predict([real_image, fake_image])
            print(batch_pil_result)

        # 4. Predict batch of file paths
        if real_image_path and fake_image_path:
            print("\n--- Predicting batch of file paths ---")
            batch_path_result = detector.predict([real_image_path, fake_image_path])
            print(batch_path_result)

        # 5. Predict mixed batch (PIL and paths)
        if real_image and fake_image_path:
             print("\n--- Predicting mixed batch (PIL/Path) ---")
             mixed_batch_result = detector.predict([real_image, fake_image_path])
             print(mixed_batch_result)

        # 6. Predict batch with invalid inputs
        print("\n--- Predicting batch with invalid inputs ---")
        invalid_batch_inputs = []
        if real_image_path: invalid_batch_inputs.append(real_image_path)
        invalid_batch_inputs.append(invalid_path) # Non-existent file
        if non_image_path: invalid_batch_inputs.append(non_image_path) # Not an image file
        if fake_image: invalid_batch_inputs.append(fake_image) # Valid PIL image
        
        if invalid_batch_inputs:
             invalid_batch_result = detector.predict(invalid_batch_inputs)
             print(invalid_batch_result)
        else:
             print("Skipping invalid batch test as dummy files could not be created.")
             
    except Exception as e:
        logging.error(f"An error occurred during detector initialization or prediction: {e}")
    
    finally:
        # Clean up dummy files
        if real_image_path and os.path.exists(real_image_path): os.remove(real_image_path)
        if fake_image_path and os.path.exists(fake_image_path): os.remove(fake_image_path)
        if non_image_path and os.path.exists(non_image_path): os.remove(non_image_path)
