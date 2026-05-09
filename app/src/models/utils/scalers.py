from app.src.models.utils.custom_scalers.robust_scaler_layer import RobustScalerLayer

def apply_robust_scaler_layer(tensor_data):
    scaler = RobustScalerLayer()
    scaler.build(tensor_data.shape)
    scaler.adapt(tensor_data)
    return scaler

scalers_executions = {
    "RobustScalerLayer": apply_robust_scaler_layer,
}