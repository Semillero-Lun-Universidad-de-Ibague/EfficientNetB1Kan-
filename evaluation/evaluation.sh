# python3 run_grad_cam.py ../../model_checkpoints/EfficientNet_10epochs_2024-11-16_best_checkpoint.pth efficientnet ../../data/Testing/meningioma_tumor/1271.jpg meningioma_efficientnet.png 
# python3 run_grad_cam.py ../../model_checkpoints/EfficientNet_ConvKAN_Mid_10epochs_2024-11-16_best_checkpoint.pth efficientnet_convkan_mid ../../data/Testing/meningioma_tumor/1271.jpg meningioma_efficientnet_convkan_mid.png
# 
# python3 run_grad_cam.py ../../model_checkpoints/VGG16_10epochs_2024-11-16_best_checkpoint.pth vgg ../../data/Testing/meningioma_tumor/1271.jpg meningioma_vgg.png
# 
# python3 run_grad_cam.py ../../model_checkpoints/ResNext_10epochs_2024-11-16_best_checkpoint.pth resnext ../../data/Testing/meningioma_tumor/1271.jpg meningioma_resnext.png

python3 run_grad_cam.py ../../model_checkpoints/ConvKAN_model_10epochs_2024-11-16_best_checkpoint.pth conv_kan ../../data/Testing/meningioma_tumor/1271.jpg meningioma_convkan.png
 

# python3 make_plots.py KAN_model_10epochs_2024-11-16_best_checkpoint ../testing/preds.json
# python3 make_plots.py ConvKAN_model_10epochs_2024-11-16_best_checkpoint ../testing/preds.json

# python3 make_plots.py EfficientNet_10epochs_2024-11-16_best_checkpoint ../testing/preds.json
# python3 make_plots.py EfficientNet_KAN_10epochs_2024-11-16_best_checkpoint ../testing/preds.json
# python3 make_plots.py EfficientNet_KAN_Mid_10epochs_2024-11-16_best_checkpoint ../testing/preds.json
# python3 make_plots.py EfficientNet_ConvKAN_Mid_10epochs_2024-11-16_best_checkpoint ../testing/preds.json
# 
# python3 make_plots.py VGG16_10epochs_2024-11-16_best_checkpoint ../testing/preds.json
# python3 make_plots.py VGG16_KAN_10epochs_2024-11-16_best_checkpoint ../testing/preds.json
# python3 make_plots.py VGG16_KAN_Mid_10epochs_2024-11-16_best_checkpoint ../testing/preds.json
# 
# python3 make_plots.py ResNext_10epochs_2024-11-16_best_checkpoint ../testing/preds.json
# python3 make_plots.py ResNext_KAN_10epochs_2024-11-16_best_checkpoint ../testing/preds.json
# python3 make_plots.py ResNext_KAN_Mid_10epochs_2024-11-16_best_checkpoint ../testing/preds.json
