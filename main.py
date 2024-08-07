import cv2
from inference.core.interfaces.camera.entities import VideoFrame
from inference import InferencePipeline
import supervision as sv
import mediapipe as mp
import numpy as np

COLOR_ANNOTATOR = sv.ColorAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()
BOX_ANNOTATOR = sv.BoxAnnotator()
True_vals = [False, False, False, False, False]
   


def show_text(annotated_frame, label, start_point, end_point):
    start_point = start_point
    end_point = end_point

    cv2.rectangle(annotated_frame, start_point, end_point, (255, 0, 157), cv2.FILLED)
    
    text = label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2

    center_x = (start_point[0] + end_point[0]) // 2
    center_y = (start_point[1] + end_point[1]) // 2

    # Get the text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, line_type)

    # Calculate the bottom-left corner of the text to center it
    text_x = center_x - text_width // 2
    text_y = center_y + text_height // 2

    # Add the text
    cv2.putText(annotated_frame, text, (text_x, text_y), font, font_scale, font_color, line_type)


def on_prediction(res: dict, frame: VideoFrame) -> None:
    global True_vals, zone
    image = frame.image

    annotated_frame = image.copy()

    result = res['output']

    if result:

        annotated_image = sv.BoxAnnotator().annotate(
        scene=annotated_frame, detections=result
        )
        annotated_image = sv.LabelAnnotator().annotate(
            scene=annotated_image, detections=result
        )

    if res['output5']:
        if not True_vals[4]:
            show_text(annotated_frame, "Put Back", (1300, 0), (1533, 130))
            True_vals[3] = True
    if res['output2']:
        if not True_vals[3]:
            show_text(annotated_frame, "Get Line", (1300, 0), (1533, 130))
            True_vals[2] = True
    if res["output6"]:
        if not True_vals[2]:
            show_text(annotated_frame, "Get Lid", (1300, 0), (1533, 130))
        True_vals[1] = True
    if res["output3"]:
        if not True_vals[1]:
            show_text(annotated_frame, "Get String", (1300, 0), (1533, 130))
        True_vals[0] = True
    if not True_vals[0]:
        show_text(annotated_frame, "Get Wheel", (1300, 0), (1533, 130))
    
    


    # Show the annotated frame
    cv2.imshow("frame", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        return

pipeline = InferencePipeline.init_with_workflow(
    video_reference="/Users/owner/Downloads/Roboflow projects/Assembly-Line/0807 (1).mp4",
    workspace_name="nathan-yan",
    workflow_id="assembly-line",
    max_fps=60,
    api_key="Zw9s4qJmfSsVpb4IerO9",
    on_prediction=on_prediction,
)

pipeline.start()
pipeline.join()
