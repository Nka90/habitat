**Dynamic object search in a 3D virtual space using an AI agent**



*Project description:*

Implementation of a 3D virtual environment and an intelligent agent capable of locating a specified, previously learned object as quickly as possible. The agent supports omnidirectional movement and camera rotation control, enabling efficient exploration of the environment. After detecting the object, the agent displays the shortest path from its initial position to the detected object and provides an image with the object highlighted from the agent’s point of view at the moment of detection.



*Main Features:*

\- Custom 3D environment

\- Synthetic dataset generation

\- Object detection using YOLOv8

\- Integration with Habitat simulator

\- Agent navigation and path visualization

\- PyQt user interface for interaction

\- Runs on CPU



*Scope:*

* Technologies Used
* Requirements
* Installation and Setup
* Usage \& Options
* Authors and Objectives
* Questions and Tips



*Technologies Used:*

Python, Habitat-Sim, PyQT5, YOLOv8 (Ultralytics), Blender (3D modeling)



*Requirements:*

The program uses components that run only on the Linux kernel. The following execution options are supported:

* Native Linux (Ubuntu 24.04 or compatible distributions)

or

* WSL (The Windows Subsystem for Linux) - more at https://learn.microsoft.com/en-us/windows/wsl/install



The program uses isolated Сonda environments. To run it, you must have one of the following installed:

* Miniconda3 (recommended) - more at https://www.anaconda.com/docs/getting-started/miniconda/install/overview

or

* Anaconda3



*Installation and Setup:*

1. Clone repository

&#x09;git clone <>

&#x09;cd habitatSearching

2\. Create environments

&#x09;Run install\_envs.sh file: ./install\_envs.sh

3\. Run the application

&#x09;Run run.sh file: ./run.sh

4\. Select a search object and click the “Go Searching” button to start your search



*Usage \& Options:*

* You can use your own 3D model in .glb format.

&#x09;Place it in the ./data/scene\_datasets folder under the name environment.glb, or specify the path to your model 	in ./exploration\_agent/config/settings.py -> class PathConfig -> the scene\_path property.

* You can use your own recognition model.

&#x09;Place it in the ./data/yolo\_model folder under the name best.pt, or specify the path to your model 	in ./exploration\_agent/config/settings.py -> class PathConfig -> yolo\_model\_path property

* You can find the frames for further use in the results, detection\_results, and exploration\_frames folders
* You can adjust the number of steps and the detection interval, thereby controlling the area coverage during exploration and the frequency of detection in ./exploration\_agent/config/settings.py -> class PathConfig.



*Authors and Objectives:*

Author: Viktoriia Korotova

Supervisor: Ing. Ondřej Budík



This project is the result of a bachelor’s thesis, České Budějovice, University of South Bohemia, Faculty of Science, Applied Informatics, 2026.



*Questions and Tips:*

* If you are using your own materials, we recommend combining your own virtual environment model with a detection model. If you use only one resource, detection is not guaranteed.
* If the agent is unable to detect an object immediately, try running the simulation several times. Since the agent is exploring the area, there is a chance that the camera may not capture a particular object.
* If you encounter any issues with the simulation, please refer to the error message displayed in the console.
* If you're having trouble with the install\_envs.sh file, try installing the environments separately using the following commands:

&#x09;conda env create -f pyqt\_app.yml - set up an environment with the necessary components for UI

&#x09;conda env create -f habitat\_env.yml - set up an environment with the necessary components for Simulator

