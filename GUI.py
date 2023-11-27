import sys
import re
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QCheckBox, \
    QMessageBox, QFileDialog, QComboBox, QRadioButton, QButtonGroup, QDesktopWidget, QHBoxLayout,QFormLayout, QVBoxLayout, QLabel, QLineEdit, QGroupBox, \
        QSpacerItem, QSizePolicy, QSpinBox
        
from PyQt5.QtGui import QFont
from deepvoxnet2.components.mirc import Mirc, Dataset, Case, Record, NiftiFileModality
from deepvoxnet2.factories.directory_structure import MircStructure
from deepvoxnet2.components.sampler import MircSampler
from deepvoxnet2.components.model import DvnModel
from deepvoxnet2.utilities.visualization_3d import visualize_output, visualize_output_overlay
import glob
import os
import nibabel as nib
import numpy as np
from deepvoxnet2 import DWI_model_dir, NCCT_model_dir, FLAIR_model_dir
from PyQt5.QtCore import Qt
import shutil
from deepvoxnet2.utilities.reorient import orient_to_lps_3d
from deepvoxnet2.utilities.activate_venv_fip import check_environment



class GUI(QWidget):
    def __init__(self):
        super().__init__()
        # Set up the main layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        #############
        ### INPUT ###
        #############
        # Create group box for column 1
        box1_groupbox = QGroupBox()
        box1_layout = QVBoxLayout()
        box1_layout.setAlignment(Qt.AlignTop)  # Align items to the top
        box1_groupbox.setLayout(box1_layout)
        main_layout.addWidget(box1_groupbox)
        
        # Label for Box 1
        box1_label = QLabel('Input\n')
        box1_label.setFont(QFont("Arial", 18, QFont.Bold))  # Set larger size and bold font
        box1_label.setAlignment(Qt.AlignHCenter)
        box1_layout.addWidget(box1_label)
        
        
        # Modality Field
        modality_label = QLabel("Modality:")
        modality_label.setFont(QFont("Arial", 14))  # Enlarge the font size
        box1_layout.addWidget(modality_label)

        self.modality_group = QButtonGroup()

        self.modality_ncct = QRadioButton("NCCT")
        self.modality_group.addButton(self.modality_ncct)
        box1_layout.addWidget(self.modality_ncct)

        self.modality_dwi = QRadioButton("DWI")
        self.modality_group.addButton(self.modality_dwi)
        box1_layout.addWidget(self.modality_dwi)

        self.modality_flair = QRadioButton("FLAIR")
        self.modality_group.addButton(self.modality_flair)
        box1_layout.addWidget(self.modality_flair)
        

        # Model Path
        model_label = QLabel("Model:")
        model_label.setFont(QFont("Arial", 14)) 
        self.model_combo = QComboBox()
        self.model_combo.addItem("Built-in NCCT model")
        self.model_combo.addItem("Built-in DWI model")
        self.model_combo.addItem("Built-in FLAIR model")
        self.model_combo.addItem("Specific model: select path!")
        self.model_combo.currentIndexChanged.connect(self.model_path_option_changed)

        self.model_entry = QTextEdit()
        self.model_entry.setFixedHeight(30)
        self.model_entry.setEnabled(False)

        self.model_button = QPushButton("Browse")
        self.model_button.clicked.connect(self.browse_model_path)
        self.model_button.setEnabled(False)

        box1_layout.addWidget(model_label)
        box1_layout.addWidget(self.model_combo)
        box1_layout.addWidget(self.model_entry)
        box1_layout.addWidget(self.model_button)

        # NIfTI File Path(s)
        nifti_label = QLabel("NIfTI file(s):")
        nifti_label.setFont(QFont("Arial", 14)) 
        self.nifti_entry = QTextEdit()
        self.nifti_entry.setFixedHeight(90)
        
        nifti_button = QPushButton("Browse")
        nifti_button.clicked.connect(self.browse_nifti_paths)

        box1_layout.addWidget(nifti_label)
        box1_layout.addWidget(self.nifti_entry)
        box1_layout.addWidget(nifti_button)
    
        
        # Patient IDs
        patient_ids_label = QLabel("(Optional) \nPatient ID(s) matching the NifTI file(s) (same order!): \nIf empty, patient ID will be the same as the name of the NifTI file.")
        patient_ids_label.setFont(QFont("Arial", 11)) 
        self.patient_ids_entry = QTextEdit()
        self.patient_ids_entry.setFixedHeight(90)
        
        box1_layout.addWidget(patient_ids_label)
        box1_layout.addWidget(self.patient_ids_entry)
        
        
        # Ground Truth Path
        ground_truth_label = QLabel("(Optional) \nGround truth file(s) matching the NifTI file(s) (same order!). \nIf no ground truth available, leave empty!:")
        ground_truth_label.setFont(QFont("Arial", 11)) 
        self.ground_truth_entry = QTextEdit()
        self.ground_truth_entry.setFixedHeight(90)
        ground_truth_button = QPushButton("Browse")
        ground_truth_button.clicked.connect(self.browse_ground_truth)

        box1_layout.addWidget(ground_truth_label)
        box1_layout.addWidget(self.ground_truth_entry)
        box1_layout.addWidget(ground_truth_button)
        
        
        # Orient Field
        orient_label = QLabel("Reorient image to LPS and ensure 3D format?")
        orient_label.setFont(QFont("Arial", 14))  # Enlarge the font size
        box1_layout.addWidget(orient_label)

        self.orient_group = QButtonGroup()

        self.orient_yes = QRadioButton("Yes")
        self.orient_group.addButton(self.orient_yes)
        box1_layout.addWidget(self.orient_yes)

        self.orient_no = QRadioButton("No: image is already 3D and LPS-oriented")
        self.orient_group.addButton(self.orient_no)
        box1_layout.addWidget(self.orient_no)
        
        
        # Additional label with formatted text
        info_label = QLabel()
        info_label.setFont(QFont("Arial", 10))
        info_label.setText("<html><body><p style='font-style: italic; font-weight: bold;'>Input image and ground truth label MUST be 3D and LPS-oriented!</p></body></html>")
        box1_layout.addWidget(info_label)
        
        
        
        # Add a vertical spacer to push the remaining fields to the top
        spacer_item = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        box1_layout.addItem(spacer_item)
        
        
        ##############
        ### OUTPUT ###
        ##############
        # Create group box for column 2
        box2_groupbox = QGroupBox()
        box2_layout = QVBoxLayout()
        box2_layout.setAlignment(Qt.AlignTop)  # Align items to the top
        box2_groupbox.setLayout(box2_layout)
        main_layout.addWidget(box2_groupbox)
        
        box2_label = QLabel('Output\n')
        box2_label.setFont(QFont("Arial", 18, QFont.Bold))  # Set larger size and bold font
        box2_label.setAlignment(Qt.AlignHCenter)
        box2_layout.addWidget(box2_label)
        
        # Output Directory
        output_label = QLabel("Output directory:")
        output_label.setFont(QFont("Arial", 14))
        self.output_entry = QTextEdit()
        self.output_entry.setFixedHeight(30)
        output_button = QPushButton("Browse")
        output_button.clicked.connect(self.browse_output_dir)
        
        # Add the output label and field directly to the main layout
        box2_layout.addWidget(output_label)
        box2_layout.addWidget(self.output_entry)
        box2_layout.addWidget(output_button)
        
        # Saving as field
        saving_label2 = QLabel("Structure to save prediction:")
        saving_label2.setFont(QFont("Arial", 14))  # Enlarge the font size
        box2_layout.addWidget(saving_label2)
        self.saving_group2 = QButtonGroup()
        self.saving_dvn2 = QRadioButton("DVN2 framework")
        self.saving_group2.addButton(self.saving_dvn2)
        box2_layout.addWidget(self.saving_dvn2)
        self.saving_simple = QRadioButton("Simple structure")
        self.saving_group2.addButton(self.saving_simple)
        box2_layout.addWidget(self.saving_simple)
        
        # Add the saving label and fields directly to the main layout
        box2_layout.addWidget(saving_label2)
        box2_layout.addWidget(self.saving_dvn2)
        box2_layout.addWidget(self.saving_simple)
        
        
        
        # Saving as field
        saving_label = QLabel("Save prediction as binary map?")
        saving_label.setFont(QFont("Arial", 14))  # Enlarge the font size
        box2_layout.addWidget(saving_label)
        self.saving_group = QButtonGroup()
        self.saving_binary = QRadioButton("Yes")
        self.saving_group.addButton(self.saving_binary)
        box2_layout.addWidget(self.saving_binary)
        self.saving_prob = QRadioButton("No: probability [0;1]")
        self.saving_group.addButton(self.saving_prob)
        box2_layout.addWidget(self.saving_prob)
        
        # Add the saving label and fields directly to the main layout
        box2_layout.addWidget(saving_label)
        box2_layout.addWidget(self.saving_binary)
        box2_layout.addWidget(self.saving_prob)
        
        # Display Field
        display_label = QLabel("Display output?")
        display_label.setFont(QFont("Arial", 14))  # Enlarge the font size
        box2_layout.addWidget(display_label)
        self.display_group = QButtonGroup()
        self.display_no = QRadioButton("No")
        self.display_group.addButton(self.display_no)
        box2_layout.addWidget(self.display_no)
        self.display_side = QRadioButton("Yes: side by side")
        self.display_group.addButton(self.display_side)
        box2_layout.addWidget(self.display_side)
        self.display_overlay = QRadioButton("Yes: overlay")
        self.display_group.addButton(self.display_overlay)
        box2_layout.addWidget(self.display_overlay)
        
        # Threshold Field
        threshold_label = QLabel("Set threshold masks:")
        threshold_label.setFont(QFont("Arial", 12))  # Enlarge the font size
        self.lower_threshold_spinbox = QSpinBox()
        self.lower_threshold_spinbox.setSpecialValueText("None")  # Set the special value text
        self.higher_threshold_spinbox = QSpinBox()
        self.higher_threshold_spinbox.setSpecialValueText("None")  # Set the special value text
        
        # Labels for threshold values
        lower_label = QLabel("Lower threshold:")
        higher_label = QLabel("Higher threshold:")
        lower_label.setVisible(False)
        higher_label.setVisible(False)
        
        # Function to toggle the visibility of the threshold field and labels
        def toggle_threshold_field():
            is_display_selected = self.display_side.isChecked() or self.display_overlay.isChecked()
            threshold_label.setVisible(is_display_selected)
            self.lower_threshold_spinbox.setVisible(is_display_selected)
            self.higher_threshold_spinbox.setVisible(is_display_selected)
            lower_label.setVisible(is_display_selected)
            higher_label.setVisible(is_display_selected)
        
        # Connect the function to the clicked signal of the radio buttons
        self.display_no.clicked.connect(toggle_threshold_field)
        self.display_side.clicked.connect(toggle_threshold_field)
        self.display_overlay.clicked.connect(toggle_threshold_field)
        
        # Initially hide the threshold field and labels
        toggle_threshold_field()
        
        # Add the labels and fields to the main layout
        box2_layout.addWidget(display_label)
        box2_layout.addWidget(self.display_no)
        box2_layout.addWidget(self.display_side)
        box2_layout.addWidget(self.display_overlay)
        box2_layout.addWidget(threshold_label)
        box2_layout.addWidget(lower_label)
        box2_layout.addWidget(self.lower_threshold_spinbox)
        box2_layout.addWidget(higher_label)
        box2_layout.addWidget(self.higher_threshold_spinbox)
        
        
        # Metrics Field
        metrics_label = QLabel("Save output metrics?")
        metrics_label.setFont(QFont("Arial", 14))  # Enlarge the font size
        box2_layout.addWidget(metrics_label)
        self.metrics_group = QButtonGroup()
        self.metrics_yes = QRadioButton("Yes")
        self.metrics_group.addButton(self.metrics_yes)
        box2_layout.addWidget(self.metrics_yes)
        self.metrics_no = QRadioButton("No")
        self.metrics_group.addButton(self.metrics_no)
        box2_layout.addWidget(self.metrics_no)
        
        
        # Run Button
        run_button = QPushButton("Run")
        run_button.setFont(QFont("Arial", 14))  # Enlarge the font size
        run_button.setFixedSize(300, 50)
        run_button.clicked.connect(self.run_process)
        box2_layout.addWidget(run_button)
        
        # Add a vertical spacer to push the remaining fields to the top
        spacer_item = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        box2_layout.addItem(spacer_item)
        


    def model_path_option_changed(self):
        selected_option = self.model_combo.currentText()
        if selected_option == "Specific model: select path!":
            self.model_entry.setEnabled(True)
            self.model_button.setEnabled(True)
        else:
            self.model_entry.setEnabled(False)
            self.model_button.setEnabled(False)

    def browse_model_path(self):
        file_dialog = QFileDialog()
        model_path = file_dialog.getExistingDirectory(self, "Select Model Path")
        self.model_entry.setText(model_path)

    def browse_nifti_paths(self):
        file_dialog = QFileDialog()
        nifti_paths, _ = file_dialog.getOpenFileNames(self, "Select NIfTI Files")
        self.nifti_entry.setText("\n".join(nifti_paths))

    def browse_ground_truth(self):
        file_dialog = QFileDialog()
        ground_truth_paths, _ = file_dialog.getOpenFileNames(self, "Select Ground Truth File")
        self.ground_truth_entry.setText("\n".join(ground_truth_paths))

    def browse_output_dir(self):
        file_dialog = QFileDialog()
        output_dir = file_dialog.getExistingDirectory(self, "Select Output Directory")
        self.output_entry.setText(output_dir)

    def run_process(self):
        
        ########################
        ### Get user choices ###
        ########################
        selected_option = self.model_combo.currentText()
        if selected_option == "Specific model: select path!":
            model_path = self.model_entry.toPlainText()
        elif selected_option == "Built-in NCCT model":
            model_path = NCCT_model_dir
        elif selected_option == "Built-in DWI model":
            model_path = DWI_model_dir
        elif selected_option == "Built-in FLAIR model":
            model_path = FLAIR_model_dir

        nifti_paths = self.nifti_entry.toPlainText().split("\n")
        
        patient_labels = self.patient_ids_entry.toPlainText().split("\n")
        if patient_labels != ['']:
            assert len(patient_labels) == len(nifti_paths), 'The number of NifTI files does not match the number of patient labels! These must be matched, or patient labels should not be specified, and the patient will be labelled identically to the name of the NifTI file.'
        
        ground_truth_paths = self.ground_truth_entry.toPlainText().split("\n")
        if ground_truth_paths != ['']:
            assert len(ground_truth_paths) == len(nifti_paths), 'The number of NifTI files does not match the number of ground truth labels. These must be matched, or ground truth labels should not be specified!'
        
        output_dir = self.output_entry.toPlainText()
        
        structure_choice = self.saving_group2.checkedButton().text()
        
        saving_choice = self.saving_group.checkedButton().text()

        display_choice = self.display_group.checkedButton().text()
        
        orient_choice = self.orient_group.checkedButton().text()



        #######################
        ### Gather the data ###
        #######################
        
        ## REORIENT
        if orient_choice == 'Yes':
            predicted_dir = os.path.join(output_dir, 'predicted')
            if not os.path.exists(predicted_dir):
                os.makedirs(predicted_dir)
            for idx in range(len(nifti_paths)):
                lps = orient_to_lps_3d(nifti_paths[idx])
                file_name = os.path.basename(nifti_paths[idx])
                extension = ''
                while True:
                    file_name, ext = os.path.splitext(file_name)
                    if ext == '':
                        break
                    extension = ext + extension
                lps_name = os.path.join(predicted_dir, file_name + '_lps' + extension)
                nib.save(lps, lps_name)
                nifti_paths[idx] = lps_name
            if ground_truth_paths != ['']:
                for idx in range(len(ground_truth_paths)):
                    lps = orient_to_lps_3d(ground_truth_paths[idx])
                    file_name = os.path.basename(ground_truth_paths[idx])
                    extension = ''
                    while True:
                        file_name, ext = os.path.splitext(file_name)
                        if ext == '':
                            break
                        extension = ext + extension
                    lps_name = os.path.join(predicted_dir, file_name + '_lps' + extension)
                    nib.save(lps, lps_name)
                    ground_truth_paths[idx] = lps_name
   
        
        dataset = Dataset("Predict")
        case_names = list()
        for case_idx in range(len(nifti_paths)):
            if patient_labels != ['']:
                case_name = patient_labels[case_idx]
                case_names.append(case_name)
            else:
                case_name = re.split(r' |/|\\', nifti_paths[case_idx])[-1].split('.')[0]
                case_names.append(case_name)
            case = Case(case_name, nifti_paths[case_idx])
            record = Record("record_0")
            record.add(NiftiFileModality("image", nifti_paths[case_idx]))
            if ground_truth_paths != ['']:
                record.add(NiftiFileModality("lesion", ground_truth_paths[case_idx]))
            case.add(record)
            dataset.add(case)
        
        
        ###############################
        ### Create output structure ###
        ###############################
        test_data = Mirc()
        test_data.add(dataset)
        test_sampler = MircSampler(test_data)
        output_structure = MircStructure(
            base_dir=output_dir,
            run_name='predicted',
            experiment_name='predicted',
            fold_i=0,
            round_i=None,
            testing_mirc=test_data
        )
        output_structure.create()
        
        ################
        ### Predict ####
        ################
        dvn_model = DvnModel.load_model(model_path)
        if ground_truth_paths != ['']:
            predictions = dvn_model.evaluate("full_val", test_sampler, output_dirs=output_structure.test_images_output_dirs)
        else:
            predictions = dvn_model.predict("full_test", test_sampler, output_dirs=output_structure.test_images_output_dirs)
        
        if saving_choice == 'Yes':
            for caseIdx in range(len(output_structure.test_images_output_dirs)):
                thisFile = glob.glob(os.path.join(output_structure.test_images_output_dirs[caseIdx], '*.nii.gz'))[0]
                pred_lesion_nifti = nib.load(thisFile)
                pred_lesion_array = pred_lesion_nifti.get_fdata()
                pred_lesion_array = np.where(pred_lesion_array<0.5, 0, 1).astype(np.uint8).squeeze()
                pred_lesion_nifti = nib.Nifti1Image(pred_lesion_array, affine = pred_lesion_nifti.affine)
                nib.save(pred_lesion_nifti, thisFile)
        
        if display_choice != 'No':
            vmin = None if self.lower_threshold_spinbox.value() == self.lower_threshold_spinbox.minimum() else self.lower_threshold_spinbox.value()
            vmax = None if self.higher_threshold_spinbox.value() == self.higher_threshold_spinbox.minimum() else self.higher_threshold_spinbox.value()
            for caseIdx in range(len(nifti_paths)):
                specific_part = re.split(r' |/|\\',output_structure.test_images_output_dirs[caseIdx])[-4]
                selected_file_paths = [(i, path) for i, path in enumerate(nifti_paths) if re.search(specific_part, path)]
                image = nib.load(selected_file_paths[0][1]).get_fdata().squeeze()
                thisFile = glob.glob(os.path.join(output_structure.test_images_output_dirs[caseIdx], '*.nii.gz'))[0]
                pred = nib.load(thisFile).get_fdata().squeeze()
                if ground_truth_paths != ['']:
                    gt = nib.load(ground_truth_paths[selected_file_paths[0][0]]).get_fdata().squeeze()
                    if display_choice == "Yes: side by side":
                        visualize_output(image, pred, gt, patient_id = specific_part, vmin = vmin, vmax = vmax)
                    elif display_choice == "Yes: overlay":
                        visualize_output_overlay(image, pred, gt, patient_id = specific_part, vmin = vmin, vmax = vmax)
                else:
                    if display_choice == "Yes: side by side":
                        visualize_output(image, pred, patient_id = specific_part, vmin = vmin, vmax = vmax)
                    elif display_choice == "Yes: overlay":
                        visualize_output_overlay(image, pred, patient_id = specific_part, vmin = vmin, vmax = vmax)
        
        
        if structure_choice == 'Simple structure':
            for caseIdx in range(len(nifti_paths)):
                if patient_labels != ['']:
                    case_name = patient_labels[caseIdx]
                else:
                    case_name = re.split(r' |/|\\', nifti_paths[caseIdx])[-1].split('.')[0]
                oldFile = glob.glob(os.path.join(output_structure.test_images_output_dirs[caseIdx], '*.nii.gz'))[0]
                newFile = os.path.join(output_dir, 'predicted', case_name) + '_predicted.nii.gz'
                os.rename(oldFile, newFile)
            
            directory_path = os.path.join(output_dir, 'predicted')
            for root, dirs, files in os.walk(directory_path, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    shutil.rmtree(dir_path, ignore_errors=True)
        
        # Show a message box indicating completion
        QMessageBox.information(self, "Process Completed", "The process has finished.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())
