# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 20:40:24 2024

@author: iffat
"""


import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QListWidget, QHBoxLayout, QScrollArea, QCheckBox, QMenu, QAction, QInputDialog,
    QMessageBox, QLineEdit
)
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QBrush, QPalette, QPen, QCursor
from PyQt5.QtCore import Qt, QRect, QPoint

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("NGI Annotation Tool")
        self.setGeometry(100, 100, 1000, 600)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.show_context_menu)


        # Initialize the image count label
        self.image_count_label = QLabel("Image Count: 0 | Track Count: 0", self)
        self.image_count_label.setAlignment(Qt.AlignCenter)

        # Set the font size, boldness, and color
        font = QFont()
        font.setPointSize(16)  # Increase the font size
        font.setBold(True)  # Make the font bold
        self.image_count_label.setFont(font)

        # Change the font color
        palette = QPalette()
        palette.setColor(QPalette.WindowText, Qt.red)  # Set the color to red (or any color you prefer)
        self.image_count_label.setPalette(palette)

        
        
        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.next_image)
        
        self.prev_button = QPushButton("Previous", self)
        self.prev_button.clicked.connect(self.prev_image)
        
        self.upload_csv_button = QPushButton("Upload CSV", self)
        self.upload_csv_button.clicked.connect(self.upload_csv)
        
        # Add the merge button
        self.merge_button = QPushButton("Merge Tracks", self)
        self.merge_button.clicked.connect(self.merge_tracks)
        

        self.image_list_widget = QListWidget(self)
        self.image_list_widget.clicked.connect(self.image_selected)

        self.class_filter_area = QScrollArea(self)
        self.class_filter_area.setWidgetResizable(True)
        self.class_filter_widget = QWidget()
        self.class_filter_layout = QVBoxLayout()
        self.class_filter_widget.setLayout(self.class_filter_layout)
        self.class_filter_area.setWidget(self.class_filter_widget)
        self.class_filter_area.setMinimumWidth(200)  # Set minimum width for class filter area

        self.confidence_filter_area = QScrollArea(self)
        self.confidence_filter_area.setWidgetResizable(True)
        self.confidence_filter_widget = QWidget()
        self.confidence_filter_layout = QVBoxLayout()
        self.confidence_filter_widget.setLayout(self.confidence_filter_layout)
        self.confidence_filter_area.setWidget(self.confidence_filter_widget)
        self.confidence_filter_area.setMinimumWidth(200)  # Set minimum width for confidence filter area

        self.track_filter_area = QScrollArea(self)
        self.track_filter_area.setWidgetResizable(True)
        self.track_filter_widget = QWidget()
        self.track_filter_layout = QVBoxLayout()
        self.track_filter_widget.setLayout(self.track_filter_layout)
        self.track_filter_area.setWidget(self.track_filter_widget)
        self.track_filter_area.setMinimumWidth(200)  # Set minimum width for track filter area

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.upload_csv_button)
        button_layout.addWidget(self.merge_button)

        filter_layout = QVBoxLayout()
        filter_layout.addWidget(QLabel("Class Filter"))
        filter_layout.addWidget(self.class_filter_area)
        filter_layout.addWidget(QLabel("Confidence Filter"))
        filter_layout.addWidget(self.confidence_filter_area)
        filter_layout.addWidget(QLabel("Track Filter"))
        filter_layout.addWidget(self.track_filter_area)

        image_list_layout = QVBoxLayout()
        image_list_layout.addWidget(self.image_list_widget)

        main_layout = QVBoxLayout()

        filter_and_image_layout = QHBoxLayout()
        filter_and_image_layout.addLayout(image_list_layout)
        filter_and_image_layout.addWidget(self.image_label, 3)
        filter_and_image_layout.addLayout(filter_layout)

        main_layout.addLayout(filter_and_image_layout)
        
        #Need to add here image count
        image_count_layout= QHBoxLayout()
        image_count_layout.addWidget(self.image_count_label)  # Use addWidget instead of addLayout
        main_layout.addLayout(image_count_layout)
        
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.image_paths = []
        self.image_names = []
        self.image_classes = []
        self.confidence_scores = []
        self.bounding_boxes = []
        self.current_rect= []
        self.filtered_image_paths = []
        self.filtered_image_names = []
        self.filtered_bounding_boxes = []
        self.filtered_image_classes = []  # Add filtered image classes
        self.filtered_confidence_scores = []  # Add filtered confidence scores
        self.current_image_index = -1
        self.folder_path = ""
        self.tracks = []
        self.predefined_classes = self.load_predefined_classes()
        self.csv_file_path = ""
        self.current_track_id= []
        self.editing_mode = False
        self.current_bbox = None
        self.mouse_start_pos = None

    # Initialize with a predefined bounding box (bbox)
        self.corner_colors = [(0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)]  # Green by default
        self.drawing = False
        self.deleting = False
        self.moving = False
        self.change_corner = False
        self.current_corner = None
        self.ix, self.iy = -1, -1
        
        self.start_point = None
        self.end_point = None
        self.image_rectangles = {}  # Dictionary to store rectangles for each image
        self.editing_new_box = False
        
    def upload_csv(self):
        self.csv_file_path = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")[0]
        if self.csv_file_path:
            self.folder_path = os.path.dirname(self.csv_file_path)  # Extract folder path from CSV file path
            self.load_image_names_from_csv(self.csv_file_path)
            self.populate_filter_options()
            self.apply_filters()

    def load_image_names_from_csv(self, csv_file_path):
        try:
            data = pd.read_csv(csv_file_path, header=None, delimiter=',')
            print("CSV loaded with comma delimiter.")
                
            print("CSV Data:")
            print(data.head())  # Print the first few rows for debugging

            # Assuming image names are in the second column (index 1)
            self.image_names = data.iloc[:, 1].tolist()
            # Assuming bounding boxes are in columns 4, 5, 6, 7
            self.bounding_boxes = data.iloc[:, 3:7].values.tolist()
            # Assuming class names are in the tenth column (index 9) and confidence scores in the eleventh (index 10)
            self.image_classes = data.iloc[:, 9].tolist()
            self.confidence_scores = data.iloc[:, 10].tolist()
            # Assuming track IDs are in the first column (index 0)
            self.tracks = data.iloc[:, 0].tolist()

        #     print("Image Names:", self.image_names)  # Debugging: Print the list of image names
        #     print("Bounding Boxes:", self.bounding_boxes)  # Debugging: Print the list of bounding boxes
        #     print("Class Names:", self.image_classes)  # Debugging: Print the list of class names
        #     print("Confidence Scores:", self.confidence_scores)  # Debugging: Print the list of confidence scores
        #     print("Tracks:", self.tracks)  # Debugging: Print the list of track IDs
        except pd.errors.EmptyDataError:
            print("Error: The selected CSV file is empty.")
        except pd.errors.ParserError:
            print("Error: Invalid CSV format. Please check the file.")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def populate_filter_options(self):
        # Initialize checkboxes lists
        self.class_checkboxes = []
        self.confidence_checkboxes = []
        self.track_checkboxes = []
        
        # Add "Select All" checkbox for Class Filter
        self.select_all_class_checkbox = QCheckBox("Select All Classes")
        self.select_all_class_checkbox.stateChanged.connect(self.select_all_classes)
        self.class_filter_layout.addWidget(self.select_all_class_checkbox)
        
        
        # Populate class filter options
        unique_classes = sorted(set(self.image_classes))
        for cls in unique_classes:
            checkbox = QCheckBox(cls)
            checkbox.stateChanged.connect(self.update_class_confidence_track_filters)  # Update on change
            self.class_filter_layout.addWidget(checkbox)
            self.class_checkboxes.append(checkbox)
        
        # Add "Select All" checkbox for Confidence Filter
        self.select_all_confidence_checkbox = QCheckBox("Select All Confidence Scores")
        self.select_all_confidence_checkbox.stateChanged.connect(self.select_all_confidences)
        self.confidence_filter_layout.addWidget(self.select_all_confidence_checkbox)
    
        # Add "Select All" checkbox for Track Filter
        self.select_all_track_checkbox = QCheckBox("Select All Tracks")
        self.select_all_track_checkbox.stateChanged.connect(self.select_all_tracks)
        self.track_filter_layout.addWidget(self.select_all_track_checkbox)
    
        # Populate initial confidence and track options
        self.update_class_confidence_track_filters()

    def select_all_classes(self, state):
        select_all = (state == Qt.Checked)
        for checkbox in self.class_checkboxes:
            checkbox.setChecked(select_all)

    def select_all_confidences(self, state):
        select_all = (state == Qt.Checked)
        for checkbox in self.confidence_checkboxes:
            checkbox.setChecked(select_all)
    
    def select_all_tracks(self, state):
        select_all = (state == Qt.Checked)
        for checkbox in self.track_checkboxes:
            checkbox.setChecked(select_all)

    def get_selected_classes(self):
        selected_classes = [checkbox.text().lower() for checkbox in self.class_checkboxes if checkbox.isChecked()]
        return selected_classes
    
    def get_selected_confidences(self):
        selected_confidences = [float(checkbox.text()) for checkbox in self.confidence_checkboxes if checkbox.isChecked()]
        return selected_confidences
    
    def get_selected_tracks(self):
        selected_tracks = [int(checkbox.text()) for checkbox in self.track_checkboxes if checkbox.isChecked()]
        return selected_tracks
    
    def update_class_confidence_track_filters(self):
        selected_classes = set(self.get_selected_classes())
        selected_tracks = set(self.get_selected_tracks())
        
        # Filter data based on selected classes and tracks
        filtered_data = pd.DataFrame({
            'image_name': self.image_names,
            'image_class': self.image_classes,
            'confidence_score': self.confidence_scores,
            'bbox': self.bounding_boxes,
            'track': self.tracks
        })
    
                
        # # Update checkboxes based on filtered data
        # self.update_confidence_and_track_checkboxes(filtered_data)
        # self.update_class_and_confidence_checkboxes(filtered_data)  # Update based on filtered data
        if selected_classes:
            filtered_data = filtered_data[filtered_data['image_class'].str.lower().isin(selected_classes)]
            self.update_confidence_and_track_checkboxes(filtered_data)  # Update based on filtered data
        elif selected_tracks:
            filtered_data = filtered_data[filtered_data['track'].isin(selected_tracks)]
            self.update_class_and_confidence_checkboxes(filtered_data)  # Update based on filtered data
        else:
            # If no filters are selected, reset checkboxes to show all options
            self.reset_class_and_confidence_checkboxes()
            self.reset_confidence_and_track_checkboxes()
            
        self.apply_filters()  # Apply filters to update the displayed images in the left bar
    
    def clear_and_populate_checkboxes(self, checkbox_list, unique_items, layout, is_confidence=False, is_track=False):
            # Store the currently selected items before clearing
            currently_selected = set([checkbox.text() for checkbox in checkbox_list if checkbox.isChecked()])
        
            # Clear existing checkboxes
            for checkbox in checkbox_list:
                layout.removeWidget(checkbox)
                checkbox.deleteLater()
            checkbox_list.clear()
            
            #  Populate new checkboxes, restoring the previous selections
            for item in unique_items:
                checkbox = QCheckBox(str(item))
                if str(item) in currently_selected:
                    checkbox.setChecked(True)  # Restore previous selection
                if is_confidence or is_track:
                    checkbox.stateChanged.connect(self.apply_filters)  # Apply filters directly
                else:
                    checkbox.stateChanged.connect(self.update_class_confidence_track_filters)
                layout.addWidget(checkbox)
                checkbox_list.append(checkbox)
    
    def update_confidence_and_track_checkboxes(self, filtered_data):
        # Update confidence checkboxes
        unique_confidences = sorted(set(filtered_data['confidence_score']))
        self.clear_and_populate_checkboxes(self.confidence_checkboxes, unique_confidences, self.confidence_filter_layout, is_confidence=True)
        
        # Update track checkboxes
        unique_tracks = sorted(set(filtered_data['track']))
        
        # Keep previously selected tracks
        previously_selected_tracks = set([checkbox.text() for checkbox in self.track_checkboxes if checkbox.isChecked()])
        
        self.clear_and_populate_checkboxes(self.track_checkboxes, unique_tracks, self.track_filter_layout, is_track=True)
        
        # Restore previous selections (preserve checked tracks)
        for checkbox in self.track_checkboxes:
            if checkbox.text() in previously_selected_tracks:
                checkbox.setChecked(True)
                
                
    def update_class_and_confidence_checkboxes(self, filtered_data):
        # Update class checkboxes
        unique_classes = sorted(set(filtered_data['image_class']))
        
        # Keep previously selected classes
        previously_selected_classes = set([checkbox.text() for checkbox in self.class_checkboxes if checkbox.isChecked()])
        
        self.clear_and_populate_checkboxes(self.class_checkboxes, unique_classes, self.class_filter_layout)
        
        # Restore previous selections (preserve checked classes)
        for checkbox in self.class_checkboxes:
            if checkbox.text() in previously_selected_classes:
                checkbox.setChecked(True)
                
        # Update confidence checkboxes
        unique_confidences = sorted(set(filtered_data['confidence_score']))
        self.clear_and_populate_checkboxes(self.confidence_checkboxes, unique_confidences, self.confidence_filter_layout, is_confidence=True)
    
  
    
    def reset_confidence_and_track_checkboxes(self):
        # Reset confidence checkboxes to show all options
        unique_confidences = sorted(set(self.confidence_scores))
        self.clear_and_populate_checkboxes(self.confidence_checkboxes, unique_confidences, self.confidence_filter_layout, is_confidence=True)
    
        # Reset track checkboxes to show all options
        unique_tracks = sorted(set(self.tracks))
        self.clear_and_populate_checkboxes(self.track_checkboxes, unique_tracks, self.track_filter_layout)
    
    def reset_class_and_confidence_checkboxes(self):
        # Reset class checkboxes to show all options
        unique_classes = sorted(set(self.image_classes))
        self.clear_and_populate_checkboxes(self.class_checkboxes, unique_classes, self.class_filter_layout)
        
        # Reset confidence checkboxes to show all options
        unique_confidences = sorted(set(self.confidence_scores))
        self.clear_and_populate_checkboxes(self.confidence_checkboxes, unique_confidences, self.confidence_filter_layout, is_confidence=True)

    
    def populate_image_list(self):
        self.image_list_widget.clear()
        self.filtered_image_paths.clear()
        self.filtered_image_names.clear()
        self.filtered_bounding_boxes.clear()
        self.filtered_image_classes.clear()  # Clear filtered image classes
        self.filtered_confidence_scores.clear()  # Clear filtered confidence scores

        if self.folder_path and self.image_names:
            for idx, image_name in enumerate(self.image_names):
                image_path = os.path.join(self.folder_path, image_name)
                if os.path.isfile(image_path):
                    self.filtered_image_paths.append(image_path)
                    self.filtered_image_names.append(image_name)
                    self.filtered_bounding_boxes.append(self.bounding_boxes[idx])
                    self.filtered_image_classes.append(self.image_classes[idx])  # Add image class
                    self.filtered_confidence_scores.append(self.confidence_scores[idx])  # Add confidence score
                    self.image_list_widget.addItem(image_name)
                else:
                    self.filtered_image_paths.append(None)
                    self.filtered_image_names.append(f"{image_name} - INVALID")
                    self.filtered_bounding_boxes.append(None)
                    self.filtered_image_classes.append(None)  # Add placeholder for invalid
                    self.filtered_confidence_scores.append(None)  # Add placeholder for invalid
    
    def display_image(self):
        if self.filtered_image_paths:
            image_path = self.filtered_image_paths[self.current_image_index]
            if image_path:
                # print(f"Displaying image: {image_path}")  # Debugging: print the image path
                # print("current image index", self.current_image_index)
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    pixmap = self.draw_bounding_boxes(pixmap, self.current_image_index)
                    self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                else:
                    self.image_label.setText("Failed to load image.")
            else:
                self.image_label.setText("Invalid image file.")

    def draw_bounding_boxes(self, pixmap, image_index):
        painter = QPainter(pixmap)
        pen = painter.pen()
        pen.setColor(QColor(255, 0, 0))
        pen.setWidth(5)
        painter.setPen(pen)
        painter.setFont(QFont('Arial', 30))
    
        def draw_corners(xmin, ymin, xmax, ymax):
            corner_radius = 10
            corners = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]
            for i, (cx, cy) in enumerate(corners):
                color = self.corner_colors[i]
                painter.setBrush(QBrush(QColor(*color)))
                painter.drawEllipse(QPoint(cx, cy), corner_radius, corner_radius)
            painter.setBrush(Qt.NoBrush)  # Reset the brush to avoid filling rectangles
        
        # Draw predefined bounding boxes
        bbox = self.filtered_bounding_boxes[image_index]
        class_name = self.filtered_image_classes[image_index]
        confidence = self.filtered_confidence_scores[image_index]
        if bbox:
            
            xmin, ymin, xmax, ymax = bbox
            print("value of xmin, ymin, xmax, ymax", xmin, ymin, xmax, ymax)
            painter.drawRect(xmin, ymin, xmax - xmin, ymax - ymin)
            painter.drawText(xmin, ymin - 10, f"{class_name} ({confidence:.2f})")
            print("Value of current image's bounding box", bbox)
            draw_corners(xmin, ymin, xmax, ymax)
            
        else:
            xmin, ymin, xmax, ymax = None, None, None, None
    
        # Get the unique identifier for the current image
        image_name = self.filtered_image_names[image_index]
        unique_image_id = (image_name, image_index)
        
        if unique_image_id in self.image_rectangles:
            for rect, label in self.image_rectangles[unique_image_id]:
                painter.drawRect(rect)
                painter.drawText(rect.left(), rect.top() - 10, label)
                rect_xmin, rect_ymin, rect_xmax, rect_ymax = rect.left(), rect.top(), rect.right(), rect.bottom()
                draw_corners(rect_xmin, rect_ymin, rect_xmax, rect_ymax)
    
        if self.drawing and self.start_point and self.end_point:
            painter.drawRect(QRect(self.start_point, self.end_point))
            drawing_xmin = min(self.start_point.x(), self.end_point.x())
            drawing_ymin = min(self.start_point.y(), self.end_point.y())
            drawing_xmax = max(self.start_point.x(), self.end_point.x())
            drawing_ymax = max(self.start_point.y(), self.end_point.y())
            draw_corners(drawing_xmin, drawing_ymin, drawing_xmax, drawing_ymax)
    
        painter.end()
        return pixmap

    
    def next_image(self):
        if self.filtered_image_paths and self.current_image_index < len(self.filtered_image_paths) - 1:
            self.current_image_index += 1
            self.image_list_widget.setCurrentRow(self.current_image_index)  # Update the list widget selection
            self.display_image()

    def prev_image(self):
        if self.filtered_image_paths and self.current_image_index > 0:
            self.current_image_index -= 1
            self.image_list_widget.setCurrentRow(self.current_image_index)  # Update the list widget selection
            self.display_image()

    def image_selected(self):
        selected_index = self.image_list_widget.currentRow()
        if selected_index >= 0 and selected_index < len(self.filtered_image_paths):
            self.current_image_index = selected_index
            self.display_image()

    def apply_filters(self):
        selected_classes = set(self.get_selected_classes())
        selected_confidences = set(self.get_selected_confidences())
        selected_tracks = set(self.get_selected_tracks())
    
        # Use pandas DataFrame for efficient filtering
        data = pd.DataFrame({
            'image_name': self.image_names,
            'image_class': self.image_classes,
            'confidence_score': self.confidence_scores,
            'bbox': self.bounding_boxes,
            'track': self.tracks
        })
    
        if selected_classes:
            data = data[data['image_class'].str.lower().isin(selected_classes)]
        if selected_confidences:
            data = data[data['confidence_score'].isin(selected_confidences)]
        if selected_tracks:
            data = data[data['track'].isin(selected_tracks)]
    
        self.filtered_image_paths = []
        self.filtered_image_names = []
        self.filtered_bounding_boxes = []
        self.filtered_image_classes = []
        self.filtered_confidence_scores = []
        self.filtered_tracks = []
        self.filtered_to_original_index = []  # New list to store mapping
        self.image_list_widget.clear()
    
        for idx, row in data.iterrows():
            image_path = os.path.join(self.folder_path, row['image_name'])
            if os.path.isfile(image_path):
                self.filtered_image_paths.append(image_path)
                self.filtered_image_names.append(row['image_name'])
                self.filtered_bounding_boxes.append(row['bbox'])
                self.filtered_image_classes.append(row['image_class'])
                self.filtered_confidence_scores.append(row['confidence_score'])
                self.filtered_tracks.append(row['track'])
                self.filtered_to_original_index.append(idx)  # Store the original index
                self.image_list_widget.addItem(row['image_name'])
            else:
                self.filtered_image_paths.append(None)
                self.filtered_image_names.append(f"{row['image_name']} - INVALID")
                self.filtered_bounding_boxes.append(None)
                self.filtered_image_classes.append(None)
                self.filtered_confidence_scores.append(None)
                self.filtered_tracks.append(None)
                self.filtered_to_original_index.append(None)  # Store None for invalid entries
    
        if self.filtered_image_paths:
            self.current_image_index = 0
            self.display_image()
    
        # Update and display the count of images
        self.update_image_count()

    def merge_tracks(self):
        selected_tracks = self.get_selected_tracks()
        
        if len(selected_tracks) < 2:
            QMessageBox.warning(self, "Selection Error", "Please select at least two tracks to merge.")
            return
        
        # Determine which track number to keep (the smaller one) and which one to merge into it
        track_to_keep = min(selected_tracks)
        tracks_to_merge = selected_tracks
        
        # Load the existing data
        data = pd.read_csv(self.csv_file_path, header=None, delimiter=',')
        
        # Update the track IDs in the data
        data.loc[data[0].isin(tracks_to_merge), 0] = track_to_keep
        
        # Sort the data to ensure it's sequential
        data = data.sort_values(by=[0, 1])
        
        # Save the updated DataFrame back to the CSV file
        data.to_csv(self.csv_file_path, header=False, index=False)
        print("Tracks merged and file saved successfully.")
        
        # Update the internal lists to reflect the changes
        self.tracks = [track_to_keep if track in tracks_to_merge else track for track in self.tracks]
        
        # Refresh the filters and display
        self.update_class_confidence_track_filters()
        self.apply_filters()
        self.update_image_count()        
         
    def update_image_count(self):
        num_images = len(self.filtered_image_paths)
        unique_tracks = len(set(self.filtered_tracks))  # Count unique track IDs
        self.image_count_label.setText(f"Image Count: {num_images} | Track Count: {unique_tracks}")

        # Make the font larger, bold, and change the color
        font = QFont()
        font.setPointSize(16)  # Increase the font size
        font.setBold(True)  # Make the font bold
        self.image_count_label.setFont(font)

        # Change the font color
        palette = QPalette()
        palette.setColor(QPalette.WindowText, Qt.red)  # Set the color to red (or any color you prefer)
        self.image_count_label.setPalette(palette)

        # Optionally, center the text
        self.image_count_label.setAlignment(Qt.AlignCenter)

    
    def load_predefined_classes(self):
        try:
            with open("predefined_classes.txt", "r") as file:
                classes = [line.strip() for line in file.readlines()]
            return classes
        except FileNotFoundError:
            print("predefined_class.txt not found.")
            return []

    def show_context_menu(self, pos):
        context_menu = QMenu(self)

        edit_labels_action = QAction("Edit Class Labels", self)
        edit_labels_action.triggered.connect(self.edit_labels)
        context_menu.addAction(edit_labels_action)

        edit_rect_action = QAction("Edit Bounding Box", self)
        edit_rect_action.triggered.connect(self.edit_rectangle)
        context_menu.addAction(edit_rect_action)
        
        add_label_action = QAction("Add Bounding Box", self)
        add_label_action.triggered.connect(self.add_label)
        context_menu.addAction(add_label_action)
        
        delete_label_action = QAction("Delete Bounding Box", self)
        delete_label_action.triggered.connect(self.delete_label)
        context_menu.addAction(delete_label_action)
        
        edit_track_class_action = QAction("Edit Class Name for Track ID", self)
        edit_track_class_action.triggered.connect(self.edit_track_class_name)
        context_menu.addAction(edit_track_class_action)


        context_menu.exec_(self.image_label.mapToGlobal(pos))

     
    def edit_labels(self):
        if not self.filtered_image_paths or self.current_image_index < 0:
            return
        
        
        image_name = self.filtered_image_names[self.current_image_index]
        unique_image_id = (image_name, self.current_image_index)
        # image_name = self.filtered_image_names[self.current_image_index]
        predefined_bbox = self.filtered_bounding_boxes[self.current_image_index]
        new_bboxes = self.image_rectangles.get(unique_image_id, [])
        
        if not predefined_bbox and not new_bboxes:
            return
        
        # Context menu to choose between editing predefined or new bounding box labels
        menu = QMenu(self)
        if predefined_bbox:
            edit_predefined_action = QAction("Edit Predefined Bounding Box Label", self)
            edit_predefined_action.triggered.connect(lambda: self.edit_predefined_label())
            menu.addAction(edit_predefined_action)
        if new_bboxes:
            edit_new_action = QAction("Edit New Bounding Box Label", self)
            edit_new_action.triggered.connect(lambda: self.edit_new_label())
            menu.addAction(edit_new_action)
        
        menu.exec_(QCursor.pos())

    def edit_predefined_label(self):
         if not self.filtered_image_paths or self.current_image_index < 0:
             return
     
         new_class, ok = QInputDialog.getItem(self, "Edit Class Label", "Select new class label:", self.predefined_classes, current=0, editable=False)
         if ok and new_class:
             self.filtered_image_classes[self.current_image_index] = new_class
             original_index = self.filtered_to_original_index[self.current_image_index]
             self.image_classes[original_index] = new_class  # Update the original list
             print("What is original index", original_index)
     
        
             # Extract bounding box information
             bounding_box = self.filtered_bounding_boxes[self.current_image_index]
             print("bounding box info from edit_predefined_label", bounding_box)
             self.save_to_csv_label(original_index, new_class, bounding_box)  # Save the changes to the CSV file
             self.display_image()

    def edit_new_label(self):
         if not self.filtered_image_paths or self.current_image_index < 0:
             return

         
         image_name = self.filtered_image_names[self.current_image_index]
         original_index= self.current_image_index
         unique_image_id = (image_name, original_index)
         
         # image_name = self.filtered_image_names[self.current_image_index] #image_name 1SC4_Camera1_07-18-19_12-17-270001.png
         print("image_name",image_name)
         new_bboxes = self.image_rectangles.get(unique_image_id, [])#new boxes [(PyQt5.QtCore.QRect(750, 359, 148, 139), 'OPHICHTHUSPUNCTICEPS-143150402')]
         print("new boxes", new_bboxes)
         if not new_bboxes:
             return

         # Prompt to select which bounding box to edit
         bbox_labels = [f"{label} ({rect.x()}, {rect.y()}, {rect.width()}, {rect.height()})" for rect, label in new_bboxes]
         print("bbox_labels", bbox_labels) #bbox_labels ['OPHICHTHUSPUNCTICEPS-143150402 (750, 359, 148, 139)']
         selected_bbox, ok = QInputDialog.getItem(self, "Select Bounding Box to Edit", "Bounding Boxes:", bbox_labels, current=0, editable=False)
         print('selected_bbox', selected_bbox) #selected_bbox PRISTIPOMOIDES-170151800
         if ok and selected_bbox:
             selected_index = bbox_labels.index(selected_bbox)
             rect, _ = new_bboxes[selected_index]

             new_class, ok = QInputDialog.getItem(self, "Edit Class Label", "Select new class label:", self.predefined_classes, current=0, editable=False)
             if ok and new_class:
                 new_bboxes[selected_index] = (rect, new_class)
                 self.image_rectangles[unique_image_id] = new_bboxes  # Update the image rectangles
                 # print("What class is in edit_new_label", new_class)
                 print("What new_bboxes[selected_index][0] is in edit_new_label", new_bboxes[selected_index][0])
                 print("What new_bboxes[selected_index][1] is in edit_new_label", new_bboxes[selected_index][1])
                 
                
                 # Extract coordinates as a list
                 
                 bounding_box= [rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height()]

                 self.update_csv(unique_image_id, bounding_box, new_class) # Save the changes to the CSV file
                 self.display_image()

    def edit_rectangle(self):
        # Create a dialog to ask which bounding box to edit
        items = ["Existing Bounding Box", "New Bounding Box"]
        item, ok = QInputDialog.getItem(self, "Edit Bounding Box", "Choose bounding box to edit:", items, 0, False)
        
        if ok and item:
            if item == "Existing Bounding Box":
                self.edit_existing_rectangle()
            elif item == "New Bounding Box":
                self.edit_new_rectangle()
    
            
    def edit_existing_rectangle(self):
        if self.filtered_bounding_boxes and 0 <= self.current_image_index < len(self.filtered_bounding_boxes):
            self.editing_mode = True
            self.editing_new_box = False
            self.current_bbox = self.filtered_bounding_boxes[self.current_image_index]
            # Update self.bounding_boxes for existing bounding box
            original_index = self.filtered_to_original_index[self.current_image_index]
            self.bounding_boxes[original_index] = self.current_bbox
            print("current bbox", self.current_bbox)
            self.image_label.setCursor(Qt.SizeAllCursor)
            
            
    
    def edit_new_rectangle(self):
        # if self.filtered_bounding_boxes and 0 <= self.current_image_index < len(self.filtered_bounding_boxes):
        image_name = self.filtered_image_names[self.current_image_index]
        unique_image_id = (image_name, self.current_image_index)
        if unique_image_id in self.image_rectangles:
            print("Unique img id now in self.image_rectangles")
            self.editing_mode = True
            self.editing_new_box = True
            
            # self.bounding_boxes[self.current_image_index] = self.filtered_bounding_boxes[self.current_image_index]
            # self.current_bbox = self.filtered_bounding_boxes[self.current_image_index]
            # original_index = self.filtered_to_original_index[self.current_image_index]
            # self.bounding_boxes[self.current_image_index] = self.current_bbox
            self.image_label.setCursor(Qt.SizeAllCursor)
        else:
            QMessageBox.information(self, "Info", "No new bounding boxes available to edit.")

    
    def add_label(self):
        self.drawing = True
        self.editing_mode = False
        # self.image_label.setCursor(Qt.ArrowCursor)
        self.image_label.setCursor(Qt.CrossCursor)
        
    def delete_label(self):
        self.deleting = True
        self.image_label.setCursor(Qt.CrossCursor)
        print("Delete label method is activated")
        
    
      
    
    def mousePressEvent(self, event):
        if self.deleting:
            x, y = event.x(), event.y()
            if event.button() == Qt.LeftButton:
                bbox = self.filtered_bounding_boxes[self.current_image_index]
                
                if bbox:
                    reply = QMessageBox.question(self, 'Delete Bounding Box',
                                                 'Do you want to delete the bounding box?',
                                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply == QMessageBox.Yes:
                        # Find the original index using the mapping
                        original_index = self.filtered_to_original_index[self.current_image_index]
                        
                        
                        print("Image name in mousepress event after deleting yes", original_index)
                        
                        if original_index is not None:
                        
                            # Update the original list
                            self.bounding_boxes[original_index] = None
    
                            # Update the filtered list
                            self.filtered_bounding_boxes[self.current_image_index] = None
        
                            self.update_csv_after_deletion(original_index)
                            self.load_image_names_from_csv(self.csv_file_path)
                            
                            # Adjust the current image index if needed
                            if self.current_image_index == len(self.filtered_image_paths)-1:
                                ## If the current index is the last index, reset to the first image
                                self.current_image_index = 0
                                print("self.current_image_index reset to 0:", self.current_image_index)
                            # If there are no images left, handle that case
                            if len(self.filtered_image_paths) == 0:
                                self.image_label.setText("No images to display.")
                            else:
                                self.populate_image_list()
                                self.display_image()    
                        
                            self.deleting = False
                            # self.display_image()
                            print("Bounding box deleted.")
                            
                    else:
                        self.deleting = False
           
            self.image_label.setCursor(Qt.ArrowCursor)
        else:
            #Existing code for drawing and editing mode:
            if self.drawing:
                self.start_point = event.pos()
                self.end_point = event.pos()
                self.current_rect = QRect(self.start_point, self.end_point)
                # self.update()
                # self.display_image()
                
            elif self.editing_mode and event.button() == Qt.LeftButton:
                
                x, y = event.x(), event.y()
                if self.editing_new_box & self.editing_mode:
                    
                    print("current_track_id", self.current_track_id)
                    image_name = self.filtered_image_names[self.current_image_index]
                    unique_image_id = (image_name, self.current_image_index)
                    if unique_image_id in self.image_rectangles:
                        for rect, label in self.image_rectangles[unique_image_id]:
                            corner_index = self.getCornerIndex((rect.left(), rect.top(), rect.right(), rect.bottom()), x, y)
                            if self.isInsideRectangle((rect.left(), rect.top(), rect.right(), rect.bottom()), x, y):
                                self.moving = True
                                self.ix, self.iy = x, y
                                self.current_rect = rect
                                self.current_label = label
                            elif corner_index is not None:
                                self.change_corner = True
                                self.current_corner = corner_index
                                self.corner_colors[corner_index] = (0, 0, 255)
                    # First, gather the updated bounding box information (new_bbox)
                    new_bbox = (self.current_rect.left(), self.current_rect.top(), 
                                self.current_rect.right(), self.current_rect.bottom())
                    # new_class= self.current_label
            
                    # Call the save function with the unique ID, updated bounding box, and class name
                    self.save_bboxInfo_for_newAddedBbox(unique_image_id, new_bbox, self.current_track_id)                  
                                
                elif not self.editing_new_box & self.editing_mode:
                    original_index = self.filtered_to_original_index[self.current_image_index]
                    if self.filtered_bounding_boxes[self.current_image_index] is None:
                        self.drawing = True
                        self.ix, self.iy = x, y
                        self.filtered_bounding_boxes[original_index] = (x, y, x, y)
                    else:                    
                        corner_index = self.getCornerIndex(self.filtered_bounding_boxes[self.current_image_index], x, y)
                        
                        if self.isInsideRectangle(self.filtered_bounding_boxes[self.current_image_index], x, y):
                            self.moving = True
                            self.ix, self.iy = x, y
                        elif corner_index is not None:
                            self.change_corner = True
                            self.current_corner = corner_index
                            self.corner_colors[corner_index] = (0, 0, 255)
                    self.bounding_boxes[original_index] = self.filtered_bounding_boxes[self.current_image_index]
                    self.save_to_csv_after_BboxEdition()
                    
        self.update()
        self.display_image()
        

    def mouseMoveEvent(self, event):
        
        
        if self.drawing:
            self.end_point = event.pos()
            self.current_rect = QRect(self.start_point, self.end_point)
            self.current_bbox = (self.current_rect.left(), self.current_rect.top(), self.current_rect.right(), self.current_rect.bottom())
    
            
        elif self.editing_mode:
            x, y = event.x(), event.y()
            original_index = self.filtered_to_original_index[self.current_image_index]
            if self.moving:
                dx, dy = x - self.ix, y - self.iy
                if self.editing_new_box:
                    self.current_rect.moveLeft(self.current_rect.left() + dx)
                    self.current_rect.moveTop(self.current_rect.top() + dy)
                    self.current_bbox = (self.current_rect.left(), self.current_rect.top(),
                                         self.current_rect.right(), self.current_rect.bottom())
                else:
                    x1, y1, x2, y2 = self.filtered_bounding_boxes[self.current_image_index]
                    self.filtered_bounding_boxes[self.current_image_index] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
                    self.current_bbox = self.filtered_bounding_boxes[self.current_image_index]
                self.ix, self.iy = x, y
            elif self.change_corner:
                if self.editing_new_box:
                    new_rect = self.updateRectangle((self.current_rect.left(), self.current_rect.top(), self.current_rect.right(), self.current_rect.bottom()), self.current_corner, x, y)
                    self.current_rect.setRect(new_rect[0], new_rect[1], new_rect[2] - new_rect[0], new_rect[3] - new_rect[1])
                    self.current_bbox = (new_rect[0], new_rect[1], new_rect[2], new_rect[3])
                else:
                    self.filtered_bounding_boxes[self.current_image_index] = self.updateRectangle(self.filtered_bounding_boxes[self.current_image_index], self.current_corner, x, y)
                    self.current_bbox = self.filtered_bounding_boxes[self.current_image_index]
                # Now, based on whether it's a new box or an existing one, call the appropriate save function
            if self.editing_new_box:
                image_name = self.filtered_image_names[self.current_image_index]
                unique_image_id = (image_name, self.current_image_index)
                new_bbox = self.current_bbox
                # if unique_image_id in self.image_rectangles:
                #     for rect, label in self.image_rectangles[unique_image_id]:
                #         self.current_label= label
                # new_class= self.current_label
                self.save_bboxInfo_for_newAddedBbox(unique_image_id, new_bbox, self.current_track_id)
          
            else:
                # Update bounding box in the original list and save changes to CSV
                self.bounding_boxes[original_index] = self.filtered_bounding_boxes[self.current_image_index]
                self.save_to_csv_after_BboxEdition()
        
        self.update()
        self.display_image()
        # self.save_to_csv_after_editing_Rectangle()
    

        
    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            if self.start_point and self.end_point:
                rect = QRect(self.start_point, self.end_point)
                
                image_name = self.filtered_image_names[self.current_image_index]
                unique_image_id = (image_name, self.current_image_index)
                
                                
                # Prompt for the class label
                new_class, ok = QInputDialog.getItem(self, "New Bounding Box", "Select class label:", self.predefined_classes, current=0, editable=False)
                    
                if ok and new_class:
                   confirmation_dialog = QMessageBox()
                   confirmation_dialog.setWindowTitle("Confirm Label")
                   confirmation_dialog.setText(f"Do you want to keep the label '{new_class}' with confidence score of 1?")
                   confirmation_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                   confirmation_dialog.setDefaultButton(QMessageBox.Yes)
                   
                   response = confirmation_dialog.exec_()
                   
                   if response == QMessageBox.Yes:
                       if unique_image_id not in self.image_rectangles:
                           self.image_rectangles[unique_image_id] = []
                           print("self image_rectangles", self.image_rectangles)
                       self.image_rectangles[unique_image_id].append((rect, f"{new_class}"))
                       print("self image_rectangles", self.image_rectangles)
                       self.save_to_csv()
                
            self.start_point = None
            self.end_point = None 
            self.image_label.setCursor(Qt.ArrowCursor)
            self.display_image()
            # self.current_rect= None
            
        elif event.button() == Qt.LeftButton:
            self.moving = False
            
            if self.change_corner:
                self.corner_colors[self.current_corner] = (0, 255, 0)  # Revert corner color to green
                self.change_corner = False
            # self.editing_mode = False
            # self.image_label.setCursor(Qt.ArrowCursor)
            # Save changes to CSV
            
    
            self.update()
        self.display_image()
        # self.save_to_csv_after_editing_Rectangle()
    
    def paintEvent(self, event):
        if self.drawing and self.start_point and self.end_point:
            self.display_image()


    def isInsideRectangle(self, rect, x, y):
        x1, y1, x2, y2 = rect
        print("Corner value of x1, y1, x2, y2- ", x1, y1, x2, y2)
        print("Mouse pressing value x,y: ", x, y)
        result= x1 <= x <= x2 and y1 <= y <= y2
        print("result value of x1 <= x <= x2 and y1 <= y <= y2", result )
        return result
    
    def getCornerIndex(self, rect, x, y):
        print("Mouse Press Event - in getCornerIndex:", x, y)
        print("top-left", (rect[0], rect[1]))
        print("top-left", (rect[2], rect[1]))
        print("top-left", (rect[0], rect[3]))
        print("top-left", (rect[2], rect[3]))
        corners = [(rect[0], rect[1]), (rect[2], rect[1]), (rect[0], rect[3]), (rect[2], rect[3])]
        min_distance = float('inf')
        nearest_corner_index = None
        for i, (cx, cy) in enumerate(corners):
            distance = (cx - x) ** 2 + (cy - y) ** 2  # squared Euclidean distance
            if distance < min_distance:
                min_distance = distance
                nearest_corner_index = i
        return nearest_corner_index
    
    
    def updateRectangle(self, rect, corner_index, x, y):
        x1, y1, x2, y2 = rect
        if corner_index == 0:
            return (x, y, x2, y2)
        elif corner_index == 1:
            return (x1, y, x, y2)
        elif corner_index == 2:
            return (x, y1, x2, y)
        elif corner_index == 3:
            return (x1, y1, x, y)
        return rect



#Update csv after editing previously existing bounding box of an image
    def save_to_csv_after_BboxEdition(self):
        existing_data = pd.read_csv(self.csv_file_path, header=None)
        print("Length of existing data", len(existing_data))
        # Ensure that the bounding box coordinates are integers and update existing_data directly
        for i, bbox in enumerate(self.bounding_boxes):
            if bbox:
                existing_data.iloc[i, 3] = int(bbox[0])  # xmin
                existing_data.iloc[i, 4] = int(bbox[1])  # ymin
                existing_data.iloc[i, 5] = int(bbox[2])  # xmax
                existing_data.iloc[i, 6] = int(bbox[3])  # ymax
            else:
                existing_data.iloc[i, 3] = None
                existing_data.iloc[i, 4] = None
                existing_data.iloc[i, 5] = None
                existing_data.iloc[i, 6] = None
    
        try:
            existing_data.to_csv(self.csv_file_path, header=False, index=False)
            print("Data successfully saved to CSV.")
        except Exception as e:
            print("Error saving to CSV:", e)


#Update csv after editing newly added bounding box of an image
    def save_bboxInfo_for_newAddedBbox(self, unique_image_id, new_bbox, current_track_id):
        try:
            # Read the existing CSV file
            data = pd.read_csv(self.csv_file_path, header=None, delimiter=',')
            
            # Unpack the image name and index from unique_image_id
            image_name, image_index = unique_image_id
            
            # Check for existing rows with matching image name and class name
            matched_row = data[(data[0] == current_track_id) &
                               (data[1] == image_name)  
                               ]
            
            if not matched_row.empty:
                # Find the matching row and update the bounding box information
                for index in matched_row.index:
                    data.at[index, 3] = new_bbox[0]  # Update X1
                    data.at[index, 4] = new_bbox[1]  # Update Y1
                    data.at[index, 5] = new_bbox[2]  # Update X2
                    data.at[index, 6] = new_bbox[3]  # Update Y2
                # print(f"Updated bounding box for {image_name} in CSV.")
            else:
                # If no matching row is found, print a message
                print(f"No matching row found for {image_name} and new_class.")
                
            # Save the updated data back to the CSV file
            data.to_csv(self.csv_file_path, header=False, index=False)
            print("CSV file updated with new bounding box information.")
    
        except Exception as e:
            print(f"Error updating bounding box information: {e}")
  
            
            
    def edit_track_class_name(self):
        # Prompt for track ID selection
        track_id, ok = QInputDialog.getItem(self, "Select Track ID", "Track ID:", [str(tid) for tid in sorted(set(self.tracks))], editable=False)
        
        if ok and track_id:
            track_id = int(track_id)
            # Prompt for new class name
            # new_class_name, ok = QInputDialog.getText(self, "New Class Name", "Enter new class name:")
            new_class_name, ok = QInputDialog.getItem(self, "New Class Name", "Select new class label:", self.predefined_classes, current=0, editable=False)
            
            if ok and new_class_name:
                # Ask for confirmation to update all or just one
                confirmation_dialog = QMessageBox()
                confirmation_dialog.setWindowTitle("Confirm Update")
                confirmation_dialog.setText(f"Do you want to update all images under Track ID {track_id} to the new class name '{new_class_name}'?")
                confirmation_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                confirmation_dialog.setDefaultButton(QMessageBox.Yes)
                response = confirmation_dialog.exec_()
                
                if response == QMessageBox.Yes:
                    # Update class names for all images under the selected track ID
                    for idx, tid in enumerate(self.tracks):
                        if tid == track_id:
                            self.image_classes[idx] = new_class_name
                    self.update_csv_track(track_id, new_class_name, update_all=True)
                else:
                    # Update class name for only the current image
                    current_idx = self.filtered_to_original_index[self.current_image_index]
                    self.image_classes[current_idx] = new_class_name
                    
                self.update_csv_track(track_id, new_class_name, update_all=True)
                # Update filtered data and refresh the display
                self.apply_filters()
            
        

    def update_csv_track(self, track_id, new_class_name, update_all):
        try:
            # Read the CSV file into a DataFrame
            data = pd.read_csv(self.csv_file_path, header=None, delimiter=',')
            
            # Debug: Print the initial state of the DataFrame
            # print("Initial CSV Data:")
            # print(data.head())
            
            # Set the column indices based on your CSV structure
            track_id_column = 0  # Adjust the column index for track ID
            image_name_column = 1  # Adjust the column index for image name
            class_name_column = 9  # Adjust the column index for class name
            
            if update_all:
                # Update class names for all images under the selected track ID
                data.iloc[:, class_name_column] = data.apply(
                    lambda row: new_class_name if row.iloc[track_id_column] == track_id else row.iloc[class_name_column], axis=1
                )
            else:
                # Update class name for only the current image
                current_image_name = self.image_names[self.filtered_to_original_index[self.current_image_index]]
                data.iloc[:, class_name_column] = data.apply(
                    lambda row: new_class_name if (row.iloc[track_id_column] == track_id and row.iloc[image_name_column] == current_image_name) else row.iloc[class_name_column], axis=1
                )
            
            # Debug: Print the updated state of the DataFrame
            print("Updated CSV Data:")
            # print(data.head())
            
            # Save the updated DataFrame back to the CSV file
            data.to_csv(self.csv_file_path, header=False, index=False)
            print("After changing class name for specific track id, file is updated")
        except Exception as e:
            print(f"Error updating csv file: {e}")

    
            
            
            
#update_csv_after_deletion: After deletion, CSV will be updated.
    
    def update_csv_after_deletion(self, original_index):
        try:
           
            data = pd.read_csv(self.csv_file_path, header=None, delimiter=',')
            # image_name = self.filtered_image_names[self.current_image_index]
            image_name = self.image_names[original_index]
            
            track_id_to_delete = self.tracks[original_index]

            # Delete the row where the image name and track ID match
            data = data[~((data.iloc[:, 1] == image_name) & (data.iloc[:, 0] == track_id_to_delete))]

            # Save the updated DataFrame back to the CSV file
            data.to_csv(self.csv_file_path, header=False, index=False)
            print("CSV file updated after deletion.")
        except Exception as e:
            print(f"Error updating CSV file: {e}")


# Maximum update and save will come when adding new label (drawing new bounding box)
    def save_to_csv(self):
         try:            
             # Load the existing data
             data = pd.read_csv(self.csv_file_path, header=None, delimiter=',')
             
             # Initialize a dictionary to keep track of track IDs per image
             image_track_ids = {}
             
             # Find the maximum track ID and increment it
             if not data.empty:
                 max_track_id = data.iloc[:, 0].max()
                 
             else:
                 max_track_id = 0
                 
             # print("unique image id before loop")
             # Iterate over each image and its rectangles
             for unique_image_id, rectangles in self.image_rectangles.items():
                 image_name, image_index = unique_image_id
                 
                 if image_name not in image_track_ids:
                     
                     image_track_ids[image_name] = max_track_id + 1
                     max_track_id += 1
                 
                 # Get the current track ID for this image
                 self.current_track_id = image_track_ids[image_name]
                 
                 
                 new_rows = []
                 for rect, label in rectangles:
                     # Get exact bounding box values
                     x, y, width, height = rect.x(), rect.y(), rect.width(), rect.height()
                     # Check if this bounding box already exists in the CSV
                     
                     # Update self.bounding_boxes with the new label
                        # Here you should update self.bounding_boxes based on the current_image_index
                     self.bounding_boxes[self.current_image_index] = (x, y, x + width, y + height, label)

                     # if not data.empty:
                     existing_boxes = data[(data[1] == image_name) & 
                                               (data[3] == rect.x()) &
                                               (data[4] == rect.y()) &
                                               (data[5] == rect.x() + rect.width()) &
                                               (data[6] == rect.y() + rect.height())]
                    
                     # if not existing_boxes.empty:
                         # Update the class name in the existing row
                     for index in existing_boxes.index:
                        data.at[index, 9] = label
                        
            
                     
                     new_row = [self.current_track_id, image_name, "", x, y, x+width, y+height, "", "", label, 1, ""]
                     new_rows.append(new_row)
                 
                 # Convert new rows to DataFrame
                 if new_rows:
                     new_data = pd.DataFrame(new_rows, columns=data.columns)
                     
         
                     # Concatenate the new data with the existing data
                     data = pd.concat([data, new_data], ignore_index=True)
             
             # Save the updated DataFrame back to the CSV file
             data.to_csv(self.csv_file_path, header=False, index=False)
             print("File saved successfully.")
        
         except Exception as e:
             print(f"Error updating CSV file: {e}")
    

# save_to_csv_label: for editing predefined label.   
    def save_to_csv_label(self, index, new_class, bounding_box):
        try:
            # Load the original CSV data
            data = pd.read_csv(self.csv_file_path, header=None, sep=',')
        
            # Update the class name and bounding box in the CSV data
            data.at[index, 9] = new_class
            data.iloc[index, 3:7] = bounding_box
        
            # Save the updated data back to the CSV file
            data.to_csv(self.csv_file_path, header=None, index=False, sep=',')
            print("CSV file updated successfully with predefined edit label.")
        except Exception as e:
            print(f"Error updating CSV file for predefined edit label: {e}") 
            

            
# This update csv is for - Add new label, then edit labels of the new added bounding box.
    def update_csv(self, unique_image_id, bounding_box, new_class):
        try:
            data = pd.read_csv(self.csv_file_path, header=None, delimiter=',')
    
            image_name, image_index = unique_image_id
            existing_boxes = data[(data[1] == image_name) & 
                                  (data[3] == bounding_box[0]) &
                                  (data[4] == bounding_box[1]) &
                                  (data[5] == bounding_box[2]) &
                                  (data[6] == bounding_box[3])]
    
            if not existing_boxes.empty:
                for index in existing_boxes.index:
                    data.at[index, 9] = new_class
            else:
                self.current_track_id = data.iloc[:, 0].max() + 1 if not data.empty else 1
                new_row = [self.current_track_id, image_name, "", bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], "", "", new_class, 1, ""]
                new_data = pd.DataFrame([new_row], columns=data.columns)
                data = pd.concat([data, new_data], ignore_index=True)
    
            data.to_csv(self.csv_file_path, header=False, index=False)
            print("File updated successfully.")
        except Exception as e:
            print(f"Error updating CSV file: {e}")



    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())