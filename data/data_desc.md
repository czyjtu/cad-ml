# File desciption

Each directory contains two files:
1. centreline.p - Pickle file containing the graph representation of the vessel.
2. representative_frame.p - Pickle file containing representative frame.
3. segmentation.png - PNG file containing binary vessel segmentation mask.

# Centreline file

In order to read the Pickle file, use the built-in pickle library in Python. You can use the following code:
```Python
import pickle

with open('centreline.p', 'rb') as file:
    vessel_graph = pickle.load(file)
```
This will get you a NetworkX ([another Python library for graph manipulation](https://networkx.org/)) graph object. The graph's nodes are points on the vessel centreline. To obtain the centreline, you will have to connect the dots :). You can do this simply and less accurately - by drawing straight lines between the nodes - or by employing some kind of an interpolation method.

The graphs were generated automatically. They were also automatically cleaned of small branches and detached segments. They can contain some erroneous nodes and edges. If you encounter any issues, please report them back. This will help me develop a better version of the algorithm.

# Representative frame
Similarly to the centreline file, this one also has to be opened via the pickle library. It contains a numpy array with the frame on top of which the vessel segmentation was performed. Read the file the same way you read the centreline file.

# Segmentation file
As mentioned above, this is a PNG file that contains a binary mask. This is the segmentation of the vessel structure. You can use matplotlib to read it into an numpy array:
```Python
import maplotlib.pyplot as plt

segmentation = plt.imread('segmentation.png')
```
The segmentations all have dimensions of 512x512 pixels and were hand-drawn by medical specialists. There is a slight chance that some masks were used for testing and do not properly depict the vessel structure. However, I did my best to eliminate them (hence the missing numbers in the 'seg' folders). I hope the data works for you!