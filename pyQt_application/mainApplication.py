from PyQt5 import QtWidgets, QtCore, QtGui
from app import Ui_MainWindow
import subprocess
import os

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.process = None
        self.visible = False
        self.processingButtons(True)
        self.navigationButtons(self.visible)
        self.visibleButtons(self.visible)
        self.actionsFromUser() 
        self.ui.MapFrame.setPixmap(QtGui.QPixmap("data/images/RoomUp.png"))
        
    def visibleButtons(self, visible):
        # Enable or disable buttons based on visibility state
        self.ui.BtnPath.setEnabled(visible)
        self.ui.BtnFramesDetection.setEnabled(visible)
        self.ui.BtnMappedObjects.setEnabled(visible)
        self.ui.BtnFramesExploration.setEnabled(visible)
        self.ui.BtnDetectionObjectFrame.setEnabled(visible)
        
    def navigationButtons(self, visible):
        # Enable or disable navigation buttons
        self.ui.BtnBackward.setEnabled(visible)
        self.ui.BtnForward.setEnabled(visible)
        
    def processingButtons(self, visible):
        self.ui.BtnGoFinding.setEnabled(visible)
        self.ui.ChoosingObject.setEnabled(visible)
        
    def actionsFromUser(self):
        #Placeholder for additional user actions
        self.ui.BtnGoFinding.clicked.connect(self.launch_simulation)
        self.ui.BtnPath.clicked.connect(self.show_agent_path)
        self.ui.BtnMappedObjects.clicked.connect(self.show_mapped_objects)
        self.ui.BtnFramesExploration.clicked.connect(lambda: self.show_frames("exploration_frames"))
        self.ui.BtnBackward.clicked.connect(lambda: self.frames_slider("backward"))
        self.ui.BtnForward.clicked.connect(lambda: self.frames_slider("forward"))
        self.ui.BtnFramesDetection.clicked.connect(lambda: self.show_frames("detection_results"))
        self.ui.ChoosingObject.currentIndexChanged.connect(lambda: self.choose_object(self.ui.ChoosingObject.currentText()))
        self.ui.BtnDetectionObjectFrame.clicked.connect(lambda: self.show_detection()) 
        
    def launch_simulation(self):
        #Launch the exploration agent when button is clicked
        self.ui.TextProcessing.setText("Processing...")
        self.visibleButtons(False)
        self.navigationButtons(False)
        self.processingButtons(False)
        self.ui.MapFrame.setPixmap(QtGui.QPixmap("data/images/RoomUp.png"))
        
        # Path to the exploration agent script
        agent_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'exploration_agent',
            'main.py'
        )
        
        chosen_object = getattr(self, "chosen_object", self.ui.ChoosingObject.currentText().lower())
        self.process = subprocess.Popen(['conda', 'run', '-n', 'habitat_yolo', 'python', agent_script,'--object', chosen_object])
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.check_process)
        self.timer.start(1000)
        
        
    def check_process(self):
        #Check if simulation finished and re-enable button
        if self.process and self.process.poll() is not None:
            self.timer.stop()
            self.visibleButtons(True)
            self.ui.BtnGoFinding.setText("Go searching")
            self.processingButtons(True)
            self.navigationButtons(True)
            object_map = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "object_location.png")
            path_map = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "2d_path_map.png")

            #analyze results and update UI accordingly
            if not os.path.exists(path_map):
                self.actionsIfErrorSimulation()
            elif not os.path.exists(object_map):
                self.actionsAfterFailedFinding()
            else:
                self.actionsAfterSuccessfulFinding()
                
            self.process = None
    
    def actionsAfterSuccessfulFinding(self):
        #Update UI to show successful finding results
        self.ui.TextProcessing.setText("Processing complete!")
        self.ui.MapFrame.setPixmap(QtGui.QPixmap("results/object_location.png"))
        
    def actionsAfterFailedFinding(self):
        #Update UI to show failed finding results
        self.ui.TextProcessing.setText("Object is not found. Try again!")
        self.visibleButtons(False)
        self.navigationButtons(False)
        self.ui.MapFrame.setPixmap(QtGui.QPixmap("results/2d_path_map.png"))
        
    def actionsIfErrorSimulation(self):
        #Update UI to show error during simulation
        self.ui.TextProcessing.setText("Error during simulation. \nPlease check the console for details.")
        self.visibleButtons(False)
        self.navigationButtons(False)
        self.ui.MapFrame.setPixmap(QtGui.QPixmap("data/images/RoomUp.png"))
        
    def show_agent_path(self):
        #Show the agent path when button is clicked
        self.ui.TextProcessing.setText("Showing agent path...")
        self.navigationButtons(False)
        self.ui.MapFrame.setPixmap(QtGui.QPixmap("results/2d_path_map.png"))
        
    def closeEvent(self, event):
        #Clean up on window close
        if self.process and self.process.poll() is None:
            self.process.terminate()
        event.accept()
        
    def show_mapped_objects(self):
        #Show the mapped objects when button is clicked
        self.ui.TextProcessing.setText("Showing mapped object...")
        self.navigationButtons(False)
        self.ui.MapFrame.setPixmap(QtGui.QPixmap("results/object_location.png"))
            
    def show_frames(self, dir):
        #Load frames and enable navigation buttons
        frames_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), dir)
        if not os.path.exists(frames_dir):
            self.ui.TextProcessing.setText("No frames found.")
            return
        self.frame_paths = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if not self.frame_paths:
            self.ui.TextProcessing.setText("No frame images in folder.")
            return
        self.current_frame = 0

        self.navigationButtons(True)

        self.update_frame_view()
        self.ui.TextProcessing.setText("Showing frames...")

    def update_frame_view(self):
        #display the current frame in the UI
        if not self.frame_paths:
            return

        pixmap = QtGui.QPixmap(self.frame_paths[self.current_frame])
        pixmap = pixmap.scaled(
            self.ui.MapFrame.width(),
            self.ui.MapFrame.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.ui.MapFrame.setPixmap(pixmap)

    def frames_slider(self, direction):
        #Buttons for next/previous frame
        if not self.frame_paths:
            return

        if direction == "backward" and self.current_frame > 0:
            self.current_frame -= 1
        elif direction == "forward" and self.current_frame < len(self.frame_paths) - 1:
            self.current_frame += 1

        self.update_frame_view()
        
    def choose_object(self, object_name):
        self.chosen_object = object_name.lower()
        
    def show_detection(self):
        self.ui.TextProcessing.setText("Detection of the chosen object...")
        self.navigationButtons(False)
        self.ui.MapFrame.setPixmap(QtGui.QPixmap("results/highest_confidence_detection.png"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.setFixedSize(1068, 661)
    window.show()
    sys.exit(app.exec_())