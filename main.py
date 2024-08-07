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

zone = np.array([[4, 11],[348, 4],[399, 1118],[9, 1130],[4, 21]])
   


def show_text(annotated_frame, label, start_point, end_point):
    start_point = start_point
    end_point = end_point

    cv2.rectangle(annotated_frame, start_point, end_point, (0, 0, 255), cv2.FILLED)
    
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

    image = frame.image

    print(res)
    annotated_frame = image.copy()

    result = res['output']
    cv2.polylines(annotated_frame, [zone], isClosed=True, color=(0, 0, 255), thickness=5)
    if result:

        annotated_image = sv.BoxAnnotator().annotate(
        scene=annotated_frame, detections=result
        )
        annotated_image = sv.LabelAnnotator().annotate(
            scene=annotated_image, detections=result
        )
    


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