
<h1 style="text-align:center;">Real-Time Object Detection with YOLOv3 and OpenCV</h1>

<h2>Features</h2>
<ul>
    <li>Real-time object detection with YOLOv3.</li>
    <li>Confidence threshold for filtering low-probability detections.</li>
    <li>Non-Maximum Suppression (NMS) to eliminate redundant bounding boxes.</li>
    <li>Dynamically displays detected object count.</li>
    <li>Highlights detected objects with bounding boxes and class labels.</li>
</ul>

<h2>Prerequisites</h2>
<ol>
    <li>Python 3.x installed on your system.</li>
    <li>Install the required Python libraries:
        <pre>pip install opencv-python numpy</pre>
    </li>
    <li>Download the YOLOv3 model files:
        <ul>
            <li><a href="https://pjreddie.com/media/files/yolov3.weights">YOLOv3 Weights</a></li>
            <li><a href="https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg">YOLOv3 Config</a></li>
            <li><a href="https://github.com/pjreddie/darknet/blob/master/data/coco.names">COCO Names</a></li>
        </ul>
    </li>
</ol>

<h2>Usage</h2>
<ol>
    <li>Clone this repository to your local system:
        <pre>git clone https://github.com/<your-username>/YOLOv3-Object-Detection.git</pre>
    </li>
    <li>Navigate to the project directory:
        <pre>cd YOLOv3-Object-Detection</pre>
    </li>
    <li>Run the script:
        <pre>python yolov3_object_detection.py</pre>
    </li>
    <li>The script will activate your webcam and display the video feed with detected objects.</li>
</ol>

<h2>Example Output</h2>
<p>Example frame showing real-time detections with bounding boxes and labels.</p>

<h2>Contributing</h2>
<p>Feel free to fork this repository and contribute by adding features like:</p>
<ul>
    <li>Support for YOLOv4 or YOLOv5.</li>
    <li>Integration with video files.</li>
    <li>Advanced visualization (e.g., detection confidence heatmaps).</li>
</ul>


<h2>Acknowledgments</h2>
<ul>
    <li><a href="https://pjreddie.com/darknet/yolo/">YOLO: You Only Look Once</a></li>
    <li><a href="https://docs.opencv.org/">OpenCV Documentation</a></li>
</ul>
