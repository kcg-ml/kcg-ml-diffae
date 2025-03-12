import io
import json
import os
from utility.minio import minio_manager
from configs.model_configs import ModelConfig
from utility.minio.progress import Progress
from utility.path import separate_bucket_and_file_path

class ModelCard:
    def __init__(self, minio_client):
        self.minio_client = minio_client
        self.model_card = None
        self.is_checkpoint= False

    def load_model_card(self, model_name: str):
        """Load a model card from Minio based on model name."""
        # Get the Minio path for the model card
        model_card_path = ModelConfig.get_model_card_minio_path(model_name)
        
        # List all objects with the given prefix
        objects = self.minio_client.list_objects("models", prefix=model_card_path)
        
        for obj in objects:
            if obj.object_name.endswith('.json'):
                response = self.minio_client.get_object("models", obj.object_name)
                self.model_card = json.load(response)
                
                self.is_checkpoint = False
                return self.model_card
        
        raise FileNotFoundError(f"No model card found for '{model_name}'")

    def load_checkpoint_model_card(self, model_name: str, model_id: int, checkpoint: int):
        """Load the checkpoint model card from Minio based on model name, model ID, and checkpoint."""
        # Get the Minio path for the checkpoint model
        model_card_path = ModelConfig.get_model_card_minio_path(model_name, model_id, checkpoint)
         
        # List all objects with the given prefix
        objects = self.minio_client.list_objects("models", prefix=model_card_path)
        
        for obj in objects:
            if obj.object_name.endswith('.json'):
                response = self.minio_client.get_object("models", obj.object_name)
                self.model_card = json.load(response)

                self.is_checkpoint = True
                return self.model_card
        
        raise FileNotFoundError(f"No checkpoint model card found for '{model_name}' with ID '{model_id}' and checkpoint '{checkpoint}'")

    def save_checkpoint_model_card(self, model_card: dict, model_id: int = None, checkpoint: int = None):
        """Save model card to a json file in Minio."""
        self.model_card= model_card 
        model_name= model_card["model_name"]
        model_uuid= model_card["model_id"] 

        # Determine the Minio path
        model_card_path = ModelConfig.get_model_card_minio_path(model_name, model_id, checkpoint) + f"-{model_uuid}.json"
        
        if minio_manager.is_object_exists(self.minio_client, "models", model_card_path):
            print(f"Model card {model_card_path} already exists. Skipping upload to prevent overwrite.")
            return  # if file already there, skip

        # Convert model card to JSON
        model_card_json = json.dumps(self.model_card, indent=4)
        model_card_data= io.BytesIO(model_card_json.encode('utf-8'))
        
        # Upload JSON to Minio
        minio_manager.upload_data(self.minio_client, "models", model_card_path, model_card_data)

        print(f"Model card successfully saved to {model_card_path}")

    def download_model(self):
        """Download a basic model based on its model card."""
        if self.model_card is None:
            raise Exception("a model card needs to be loaded first")
        
        model_name= self.model_card["model_name"]

        for file_info in self.model_card['model_file_list']:
            bucket, minio_file_path = separate_bucket_and_file_path(file_info['minio_file_path'])
            local_file_path = file_info['file_path']

            print(f"Downloading the model to the local pth: {local_file_path}")
            local_dir = os.path.dirname(local_file_path)
            os.makedirs(local_dir, exist_ok=True)
            
            if self.is_checkpoint:
                self.minio_client.fget_object(bucket, minio_file_path, local_file_path, progress=Progress())
            else:
                minio_manager.download_from_minio(self.minio_client, bucket, minio_file_path, local_file_path)
        
        model_path = ModelConfig.get_model_local_path(model_name)
        return model_path
    
    # Getter methods
    def get_model_name(self):
        """Get the model name from the model card."""
        if self.model_card is None:
            raise Exception("the model card needs to be downloaded first")

        return self.model_card.get('model_name')

    def get_model_id(self):
        """Get the model ID from the model card."""
        if self.model_card is None:
            raise Exception("the model card needs to be downloaded first")
        
        return self.model_card.get('model_id')

    def get_file_list(self):
        """Get the list of files and their details from the model card."""
        if self.model_card is None:
            raise Exception("the model card needs to be downloaded first")
        
        return self.model_card.get('model_file_list', [])

    def get_model_size(self):
        """Get the total model size from the model card."""
        if self.model_card is None:
            raise Exception("the model card needs to be downloaded first")
        
        return self.model_card.get('model_size')
