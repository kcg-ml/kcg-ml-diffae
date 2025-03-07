
class ModelConfig:
    """Configuration class for models and model paths."""
    
    MODEL_CHECKPOINTS = {
        "DIFFAE_MODEL": "diffae",
    }
    
    MODELS = {
    }

    @staticmethod
    def is_valid_model(model_name: str) -> bool:
        """Check if the model name is in the list of available models."""
        return model_name in ModelConfig.MODELS.values()

    @staticmethod
    def is_valid_checkpoint_model(model_name: str) -> bool:
        """Check if the model name is in the list of available trained models."""
        return model_name in ModelConfig.MODEL_CHECKPOINTS.values()
    
    @staticmethod
    def get_model_card_minio_path(model_name: str, model_id: int = None, checkpoint: int = None) -> str:
        """
        Get the Minio path for the model card.
        """
        if ModelConfig.is_valid_model(model_name):
            return f"model-cards/{model_name}"
        
        elif ModelConfig.is_valid_checkpoint_model(model_name):
            if model_id is None or checkpoint is None:
                raise ValueError("model_id and checkpoint must be provided for checkpoint models.")
            
            return f"{model_name}/trained_models/{str(model_id).zfill(4)}/model-cards/{model_name}-{str(checkpoint).zfill(4)}"
        
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized as one of the available models.")

    @staticmethod
    def get_model_local_path(model_name: str) -> str:
        """Get the local path where the model will be stored."""
        if ModelConfig.is_valid_model(model_name) or ModelConfig.is_valid_checkpoint_model(model_name):
            return f"models/{model_name}"
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized as one of the available models.")