# references and credits 
# pothole detection - https://learnopencv.com/tag/pothole-detection/
# depth prediction  - https://learnopencv.com/depth-anything/

# pip install ultralytics # for yolo v8

# imports 
import cv2 
import numpy as np 
import time 
from ultralytics import YOLO
import utils_depth  

# helper function to stack frames vertically and horizontally 
def get_stacked_frames(f1, f2, f3, f4):
    hstacked1 = np.hstack((f1,f2))
    hstacked2 = np.hstack((f3,f4))
    all_stacked = np.vstack((hstacked1, hstacked2))
    return all_stacked

# yolo object detection/prediction 
def get_yolo_preds(frame):
    outputs = model.predict(source=frame)
    results = outputs[0].cpu().numpy()
    rects = []
    for i, (x1,y1,x2,y2) in enumerate(results.boxes.xyxy):
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        rects.append( ( (x1,y1), (x2,y2) ) )
    return rects

# draw yolo rects on frame 
def draw_yolo_rects(frame, rects):
    for rect in rects: 
        pt1, pt2 = rect[0], rect[1] # pt1/pt2 are in (x,y) format , already converted to int format 
        cv2.rectangle( frame, pt1, pt2, color=(0,0,255), thickness=2)
    return frame 

# compute pothole depths from depth map and pothole rects
# pothole depths are obtained by averaging across all/per-pixel depths for each pothole rect
def get_pothole_depths(depth, rects):
    pothole_depths = []
    for p1,p2 in rects: 
        pothole_roi = depth[p1[1]:p2[1], p1[0]:p2[0]]
        avg_dep = np.average(pothole_roi)
        pothole_depths.append(avg_dep)
    return pothole_depths 

# draw yolo rects on frame 
def draw_pothole_rects_depths(frame, pothole_depths, rects):
    for pothole_depth, (pt1, pt2) in zip(pothole_depths, rects): # pt1/pt2 are in (x,y) format , already converted to int format 
        cv2.rectangle( frame, pt1, pt2, color=(0,0,255), thickness=2)
        cv2.putText(frame, f'{pothole_depth:.2f}', pt1, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3 )
    return frame

# defining parameters
input_video_filepath  = './assets/road_with_pothole_Apr3.mp4'
output_video_filepath = './assets/output_v2_road_with_pothole_Apr3.mp4'

# getting input video handler and properties 
cap = cv2.VideoCapture(input_video_filepath)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('input video fps: {}, frame_width: {}, frame_height: {}'.format(fps, frame_width, frame_height))

# setting output video writer 
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID for mp4
output = cv2.VideoWriter(output_video_filepath, fourcc, fps, (2*frame_width, 2*frame_height))

# yolo model for pothole detection 
model_filepath = 'best.pt'
model = YOLO(model_filepath)

# depth-anything class for depth estimation
da = utils_depth.CustomDepthAnything()

# processing each frame and writing to output 
start = time.time()
frames_processed = 0
while(cap.isOpened()):
    frames_processed += 1
    if frames_processed < 100 or frames_processed > 200: 
        continue
    ret, orig_frame = cap.read()
    if not ret :
        print('no more frames to process. breaking from while loop') 
        break 
    
    # pothole object detection 
    rects = get_yolo_preds(orig_frame) # rects is list of tuples (pt1,pt2) where each pt1/pt2 is in (x,y) format 
    yolo_frame = draw_yolo_rects(orig_frame.copy(), rects) # draw detected rects on frame 
    
    # depth prediction 
    depth = da.get_depth_pred(orig_frame)
    norm_depth = da.get_normalised_depth(depth)
    depth_frame = da.draw_depth_map(norm_depth)

    # fused image
    pothole_depths = get_pothole_depths(255-norm_depth, rects) # 255-norm_depth so that closer distances are smaller in value
    fused_frame    = draw_pothole_rects_depths(orig_frame.copy(), pothole_depths, rects)

    # stacked frames with naming 
    font_scale = 2
    color = (0,0,255)
    origin = (50,80)
    orig_frame = cv2.putText(orig_frame, 'orignal', origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color ,2)
    yolo_frame = cv2.putText(yolo_frame, 'pothole detection', origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color ,3)
    depth_frame = cv2.putText(depth_frame, 'depth estimation', origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color ,3)
    fused_frame = cv2.putText(fused_frame, 'fused pothole with depth', origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color ,3)
    stacked_frames = get_stacked_frames(orig_frame, yolo_frame, depth_frame, fused_frame)
    output.write(stacked_frames)

    cv2.imshow('output', stacked_frames)
    if cv2.waitKey(1) == ord('q'):
        break

total_elapsed = time.time() - start 
print(f'total elapsed time: {total_elapsed:.2f}, frames_processed: {frames_processed}, avg elapsed_time: {total_elapsed/frames_processed}')

# Release all resources
cap.release()
output.release()
cv2.destroyAllWindows()

print("Done!")
