from ultralytics import YOLO
import cv2
import numpy as np

def draw_analysis_region(frame, bbox):
    """Vẽ vùng phân tích mới (10%-50% chiều cao) lên frame"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Lấy vùng cầu thủ
    height = y2 - y1
    width = x2 - x1
    
    # Tính toán vùng phân tích (10%-50% chiều cao)
    analysis_y1 = y1 + int(height * 0.1)  # 10% từ trên
    analysis_y2 = y1 + int(height * 0.5)  # 50% từ trên
    analysis_x1 = x1  # Giữ nguyên chiều rộng
    analysis_x2 = x2
    
    # Vẽ bbox gốc (màu xanh lá)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Vẽ vùng phân tích
    overlay = frame.copy()
    cv2.rectangle(overlay, (analysis_x1, analysis_y1), (analysis_x2, analysis_y2), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Vẽ viền vùng phân tích (màu đỏ đậm)
    cv2.rectangle(frame, (analysis_x1, analysis_y1), (analysis_x2, analysis_y2), (0, 0, 255), 2)
    
    # Thêm text mô tả
    cv2.putText(frame, "Analysis Region", (analysis_x1, analysis_y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return frame

def process_video_with_analysis():
    """Xử lý video và hiển thị vùng phân tích"""
    model = YOLO("../training/best_2.pt")
    
    # Đọc video
    cap = cv2.VideoCapture('../input_video/A_3.mp4')
    
    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {fps} FPS")
    
    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./output/analysis_visualization.mp4', fourcc, fps, (width, height))
    
    frame_count = 0
    player_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Detect objects
        results = model.predict(frame, verbose=False)
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # Lấy thông tin box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Lấy class name từ model
                class_names = results[0].names
                class_name = class_names.get(cls, f"class_{cls}")
                
                if conf > 0.3:
                    # Màu sắc theo class
                    if class_name.lower() in ['player', 'person']:
                        color = (0, 255, 0)  # Xanh lá cho player
                        player_count += 1
                        # Vẽ vùng phân tích cho player
                        frame = draw_analysis_region(frame, [x1, y1, x2, y2])
                    elif class_name.lower() in ['ball', 'football']:
                        color = (255, 0, 0)  # Xanh dương cho ball
                    elif class_name.lower() in ['referee', 'ref']:
                        color = (0, 255, 255)  # Vàng cho referee
                    else:
                        color = (255, 255, 255)  # Trắng cho class khác
                    
                    # Vẽ bbox cho tất cả detections
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Thêm thông tin class và confidence
                    cv2.putText(frame, f"{class_name}: {conf:.2f}", 
                               (int(x1), int(y1) - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Players detected: {len(results[0].boxes) if results[0].boxes is not None else 0}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Completed! Processed {frame_count} frames")
    print(f"Total player detections: {player_count}")
    print(f"Output saved: ./output/analysis_visualization.mp4")
    
    cap = cv2.VideoCapture('../input_video/A_3.mp4')
    ret, first_frame = cap.read()
    if ret:
        results = model.predict(first_frame, verbose=False)
        print("\n=== MODEL CLASS NAMES ===")
        class_names = results[0].names
        print("Available classes:")
        for cls_id, cls_name in class_names.items():
            print(f"  Class {cls_id}: {cls_name}")
        
        print("\n=== FIRST FRAME DETECTION DETAILS ===")
        print(f"Number of detections: {len(results[0].boxes) if results[0].boxes is not None else 0}")
        
        if results[0].boxes is not None:
            class_counts = {}
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = class_names.get(cls, f"class_{cls}")
                
                # Đếm số lượng từng class
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                
                # Chỉ tính vùng phân tích cho player
                if class_name.lower() in ['player', 'person']:
                    height = y2 - y1
                    analysis_y1 = y1 + int(height * 0.1)
                    analysis_y2 = y1 + int(height * 0.5)
                    analysis_height = analysis_y2 - analysis_y1
            
            print("=== CLASS SUMMARY ===")
            for class_name, count in class_counts.items():
                print(f"{class_name}: {count} detections")
    
    cap.release()

if __name__ == "__main__":
    import os
    os.makedirs('./output', exist_ok=True)
    process_video_with_analysis()