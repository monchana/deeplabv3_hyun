from inference import evaluate_iou
from predict_syn import predict

input_path = ''
checkpoint_path = ''
output_path = ''
predict(input_path, checkpoint_path, output_path)

diff_pred_path = ''
gt_path = output_path
result = evaluate_iou(diff_pred_path, gt_path)
