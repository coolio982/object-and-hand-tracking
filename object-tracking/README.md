# Object Tracking
Repository for evaluating object tracking methods.<br> 
Ensure you are in this directory before running the files below. 

# Directory
<pre>
│  DataCollection.py
│  DetectionEvaluation.ipynb
│  ObjectTracking.py
│  
├─Images
│  ├─DepthImage
│  │  │  frame_1686210461.jpg
│  │  │  ...
│  │  └─ frame_1686211019.jpg
│  │          
│  └─groundTruth.csv

</pre>
### DataCollection.py
This is a program to collect data samples of object locations.<br>
A colour video feed will show up. To save object locations, click on the centre of the object you want to detect using your mouse. 

### DetectionEvaluation.ipynb
This is a jupyter notebook that evaluates the recognition accuracy and speed of 3 object recognition algorithms.

### ObjectTracking.py
A sample script that can visualise the effectiveness of various tracking methods. Three methods are provided:
* Centre distances
* Lukas-Kanade
* Farneback

### Images
The following files are found in this directory:
* Sample evaluation data(DepthImage/)
* Label data(groundTruth.csv)

 