import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from tensorflow.keras import layers as tfl

@dataclass
class InputsStruct:
    categorical: List[str]
    numerical: List[str]

@dataclass
class SplitStruct:
    train: float
    val: float
    test: float

@dataclass
class HyperparametersStruct:
    lr: float
    batch_size: int = 32

@dataclass
class ScalersStruct:
    inputs: List[str]
    instance: Optional[tfl.Layer] = None

    @staticmethod
    def parse_scalers(scalers_dict: dict) -> Dict[str, 'ScalersStruct']:
        """
        Recibe el bloque 'scalers' del JSON y devuelve un diccionario de structs.
        Ej: {"RobustScalerLayer": {"inputs": [...]}} -> {"RobustScalerLayer": ScalersStruct(...)}
        """
        if not scalers_dict:
            return {}
            
        parsed = {}
        for scaler_name, config in scalers_dict.items():
            parsed[scaler_name] = ScalersStruct(inputs=config.get("inputs", []))
            
        return parsed


@dataclass
class ConfigModel:
    inputs_lags: int
    outputs_horizons: int
    inputs: InputsStruct
    scalers: Dict[str, ScalersStruct]
    outputs: list[str]
    base_model: str
    hyperparameters_init: HyperparametersStruct
    split: SplitStruct
    models: Dict[str, Any]
    sequence_block_type: str = "basic_lstm"
    sequence_block_params: Dict[str, Any] = None
    mlp_block_params: Dict[str, Any] = None

    @classmethod
    def from_dict(cls, data: dict):
        # Extraer configuración específica del MainModel
        main_model_cfg = data.get("models", {}).get("main_model", {})
        
        return cls(
            inputs_lags=data["inputs_lags"],
            outputs_horizons=data.get("outputs_horizons", 1),
            inputs=InputsStruct(**data["inputs"]),
            scalers=ScalersStruct.parse_scalers(data.get("scalers", {})),
            outputs=data["outputs"],
            base_model=data.get("base_model", ""),
            hyperparameters_init=HyperparametersStruct(**data["hyperparameters_init"]),
            split=SplitStruct(**data["split"]),
            models=data.get("models", {}),
            sequence_block_type=main_model_cfg.get("sequence_block_type", "basic_lstm"),
            sequence_block_params=main_model_cfg.get("sequence_block_params", {}),
            mlp_block_params=main_model_cfg.get("mlp_block_params", {})
        )



    def __str__(self):
        
        cfg_dict = asdict(self)
        return json.dumps(cfg_dict, indent=4, ensure_ascii=False)