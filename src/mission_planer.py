import sys
import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (QInputDialog, QApplication, QCheckBox, QDoubleSpinBox, QHBoxLayout, QGridLayout, QGroupBox, QMenu, QPushButton, QRadioButton, QVBoxLayout, QWidget, QMessageBox, QLabel, QErrorMessage, QLineEdit)
from traj import *
from Boustrophedon import *
from scipy.spatial import distance


class Terrain(QWidget):
    def __init__(self, data={"fond":"terrain.png", "type":"poly", "poly":[], "obstacles":[], "path":[], "d":5, "l":8, "org":(10, 10), 'scale':0.27, 'coords':'48.418075 -4.474395', 's':1}):
        QWidget.__init__(self)
        self.setWindowTitle("Terrain")
        self.image = QtGui.QImage('fond/'+data["fond"])
        width, height = self.image.width(), self.image.height()
        self.resize(width, height)
        self.setFixedSize(width, height)
        #self.setStyleSheet("background-color: black;")
        self.pal = QtGui.QPalette()
        self.pal.setBrush(QtGui.QPalette.Background, QtGui.QBrush(self.image))
        self.setAutoFillBackground(True)
        self.penBlanc = QtGui.QPen(QtGui.QColor(255,255,255))                # set lineColor
        self.penBlanc.setWidth(2)
        self.penV = QtGui.QPen(QtGui.QColor(139, 58, 119))                # set lineColor
        self.penV.setWidth(2)
        self.penRouge = QtGui.QPen(QtGui.QColor(255, 0, 0))                # set lineColor
        self.penRouge.setWidth(2)
        self.penVert = QtGui.QPen(QtGui.QColor(0, 255, 0))                # set lineColor
        self.penVert.setWidth(2)                                                 # set lineWidth
        self.brush = QtGui.QBrush(QtGui.QColor(255,255,255,255))        # set fillColor

        self.wps = []
        self.np_wps = np.empty((1, 2))

        #self.polygon = QtGui.QPolygonF() 

        #chargement de data :
        self.data = data
        self.boustro = (False, 0, 0)
        #self.bph = Boustrophedon(self.data['poly'], self.data['d'], self.data['obstacles'])

        # bools de contrôle
        self.bool_poly = False
        self.bool_obstacles = False
        self.set_org = False
        self.choose_scale = False
        self.bool_path = False

        #variables intermédiaires
        self.unfinished = []
        self.scaleA = (0, 0)
        self.scaleB = (0, 0)
        
        self.update()

        
    def mousePressEvent(self, e):
        x, y = e.x(), e.y()

        # moving a point
        for pt in self.data['poly']:
            if distance.euclidean((x, y), pt) < 5:
                print("moving")

        if self.choose_scale:
            self.scaleA = (x, y)

        
    def mouseReleaseEvent(self, e):
        x, y = e.x(), e.y()


        # draw poly:
        if self.bool_poly:
            if len(self.unfinished)>=3:
                if distance.euclidean((x, y), self.unfinished[0]) < 5:
                    self.bool_poly = False
                    self.data['poly'] = self.unfinished
                    self.unfinished = []
            if self.bool_poly:
                self.unfinished.append((x, y))

        elif self.bool_obstacles:
            if len(self.unfinished)>=3:
                if distance.euclidean((x, y), self.unfinished[0]) < 5:
                    self.bool_obstacles = False
                    self.data['obstacles'].append(self.unfinished)
                    self.unfinished = []
            if self.bool_obstacles:
                self.unfinished.append((x, y))

        elif self.set_org:
            self.data['org'] = (x, y)
            self.set_org = False

        elif self.choose_scale:
            self.choose_scale = False
            self.scaleB = (x, y)
            d_pixel = distance.euclidean(self.scaleA, self.scaleB)
            d_true, okPressed = QInputDialog.getInt(self, "Echelle","Distance(m) :", d_pixel*self.data['scale'], 0, 200, 1)
            if okPressed:
                self.data['scale'] = d_true/d_pixel
                print(self.data['scale'])
        
        elif self.bool_path:
            self.unfinished.append((x, y))

        self.update()

    def mouseDoubleClickEvent(self, e):
        x, y = e.x(), e.y()
        if self.bool_path:
            self.unfinished.append((x, y))
            self.data['path'] = self.unfinished
            self.unfinished = []
            self.bool_path = False


        self.update()


    def compute(self):
        if self.data['type'] == 'poly' or self.data['type'] == 'obs':
            print('s/s : ', self.data['s']/self.data['scale'])
            b = Boustrophedon(self.data['poly'], self.data['d']/self.data['scale'], self.data['l']/self.data['scale'], self.data['s']/self.data['scale'], self.data["obstacles"])
            try:
                b.decomposition()
                b.fuse_cells(10)

                trajs = []

                all_wps = []
                for cell in b.Lp:
                    wps = cell.waypoints
                    wps_array = np.zeros((len(wps), 2))
                    for i in range(len(wps)):
                        wps_array[i, 0] = wps[i].x
                        wps_array[i, 1] = wps[i].y
                    all_wps.append(wps_array)
                all_wps_array = all_wps[0]
                for a in all_wps[1:]:
                    all_wps_array = np.vstack((all_wps_array, a))

                traj_wp, traj_def = triTraj(all_wps_array, self.data)
                trajs.append((traj_wp, traj_def))
                self.boustro = (True, b, trajs)

            except Exception as e:
                print(e)
                QMessageBox.about(self, "Error", "Decomposition failed ! Do one of the following: \n- try to add more vertices describing the outer area / obstacles \n- try to use more convex / simpler polygons")

        else:
            wps = self.data['path']
            wps_array = np.zeros((len(wps), 2))
            for i in range(len(wps)):
                wps_array[i, 0] = wps[i][0]
                wps_array[i, 1] = wps[i][1]
            traj_wp, traj_def = triTraj(wps_array, self.data, traj_type='path')
            trajs = [(traj_wp, traj_def)]
            b = 0
            self.boustro = (True, b, trajs)

        print('File saved')

        self.update()

    ############################# call_backs #################

    def drawBoustro(self, boustro):
        pass

    def def_origin(self):
        self.set_org = True

    def drawPoly_cb(self):
        self.data["type"] = "poly"
        self.data["path"] = []
        self.data["poly"] = []
        self.data["obstacles"] = []
        self.boustro = (False, 0, 0)
        self.bool_poly = True
        self.bool_path = False

    def drawObstacle_cb(self):
        self.data["type"] = "obs"
        self.data["path"] = []
        self.boustro = (False, 0, 0)
        self.bool_obstacles = True
        self.bool_path = False

    def scale_cb(self, a):
        self.choose_scale = True

    def digit_boxes_cb(self, d, l, s):
        self.data['d'] = d
        self.data['l'] = l
        self.data['s'] = s

    def path_cb(self):
        self.data["type"] = "path"
        self.data["poly"] = []
        self.boustro = (False)
        self.data["path"] = []
        self.bool_path = True
        self.bool_poly = False

    def coords_cb(self, coords):
        self.data['coords'] = coords

    ###########################################################

    def paintEvent(self, event):
        self.setPalette(self.pal)
        painter = QtGui.QPainter(self)
        painter.setRenderHints(QtGui.QPainter.Antialiasing)

        #affichage origine
        org = self.data['org']
        l_arrow = 20
        A, B, C = QtCore.QPoint(org[0], org[1]), QtCore.QPoint(org[0] + l_arrow, org[1]), QtCore.QPoint(org[0], org[1]-l_arrow)
        painter.setPen(self.penVert)
        painter.drawLine(A, B)
        painter.setPen(self.penRouge)
        painter.drawLine(A, C)

        # affichage poly et path:
        if not self.boustro[0]:
            painter.setPen(self.penBlanc)
            if self.data['type'] == 'poly':
                data = self.data['poly']
                k = -1

                for wp in data:
                    painter.drawEllipse(QtCore.QPoint(wp[0], wp[1]), 5, 5)

                if len(data) >= 2:
                    for i in range(k, len(data) - 1):
                        x = data[i]
                        x = QtCore.QPoint(x[0], x[1])
                        y = data[i + 1]
                        y = QtCore.QPoint(y[0], y[1])
                        painter.drawLine(x, y)
            elif self.data['type'] == 'obs':
                data_obs = self.data['obstacles']
                data_poly = self.data['poly']
                k = -1

                for wp in data_poly:
                    painter.drawEllipse(QtCore.QPoint(wp[0], wp[1]), 5, 5)

                if len(data_poly) >= 2:
                    for i in range(k, len(data_poly) - 1):
                        x = data_poly[i]
                        x = QtCore.QPoint(x[0], x[1])
                        y = data_poly[i + 1]
                        y = QtCore.QPoint(y[0], y[1])
                        painter.drawLine(x, y)
                for obs in data_obs:
                    for wp in obs:
                        painter.drawEllipse(QtCore.QPoint(wp[0], wp[1]), 5, 5)
                    if len(obs) >= 2:
                        for i in range(k, len(obs) - 1):
                            x = obs[i]
                            x = QtCore.QPoint(x[0], x[1])
                            y = obs[i + 1]
                            y = QtCore.QPoint(y[0], y[1])
                            painter.drawLine(x, y)
            else:
                data = self.data['path']
                k = 0

                for wp in data:
                    painter.drawEllipse(QtCore.QPoint(wp[0], wp[1]), 5, 5)

                if len(data) >= 2:
                    for i in range(k, len(data) - 1):
                        x = data[i]
                        x = QtCore.QPoint(x[0], x[1])
                        y = data[i + 1]
                        y = QtCore.QPoint(y[0], y[1])
                        painter.drawLine(x, y)


        # affichage dessin en cours
        if self.bool_poly or self.bool_path or self.bool_obstacles:
            painter.setPen(self.penV)
            for wp in self.unfinished:
                painter.drawEllipse(QtCore.QPoint(wp[0], wp[1]), 5, 5)
            if len(self.unfinished)>=2:
                for i in range(len(self.unfinished)-1):
                    x = self.unfinished[i]
                    x = QtCore.QPoint(x[0], x[1])
                    y = self.unfinished[i+1]
                    y = QtCore.QPoint(y[0], y[1])
                    painter.drawLine(x, y)


        if self.boustro[0]:
            if self.data['type'] == 'poly' or self.data['type'] == 'obs':
                painter.setPen(self.penBlanc)
                B = self.boustro[1]
                for cell in B.Lp:
                    wps = cell.waypoints
                    for i in range(len(wps)-1):
                        a, b = QtCore.QPoint(wps[i].x, wps[i].y), QtCore.QPoint(wps[i+1].x, wps[i+1].y)
                        painter.drawLine(a, b)
                painter.setPen(self.penRouge)
                n = len(B.outer_raw.p.vertices)
                for i in range(n):
                    a, b = QtCore.QPoint(B.outer_raw.p.vertices[i].x, B.outer_raw.p.vertices[i].y), QtCore.QPoint(B.outer_raw.p.vertices[(i+1)%n].x, B.outer_raw.p.vertices[(i+1)%n].y)
                    painter.drawLine(a, b)
                for obs in B.obstacles_raw:
                    n = len(obs.p.vertices)
                    for i in range(n):
                        a, b = QtCore.QPoint(obs.p.vertices[i].x, obs.p.vertices[i].y), QtCore.QPoint(obs.p.vertices[(i+1)%n].x, obs.p.vertices[(i+1)%n].y)
                        painter.drawLine(a, b)

            
            painter.setPen(self.penV)
            for traj in self.boustro[2]:
                '''
                for i in range(traj.shape[0] - 1):
                    a, b = QtCore.QPoint(traj[i, 0], traj[i, 1]), QtCore.QPoint(traj[i+1, 1], traj[i+1, 1])
                    painter.setPen(self.penV)
                    painter.drawLine(a, b)'''
                    
                traj_def = traj[1]
                wps_traj = traj[0]
                for i in range(len(traj_def)):
                    if traj_def[i][0] == 'line':
                        p1 = QtCore.QPoint(wps_traj[i, 0], wps_traj[i, 1])
                        p2 = QtCore.QPoint(wps_traj[i+1, 0], wps_traj[i+1, 1])
                        painter.drawLine(p1, p2)
                    else:
                        _, x, y, a1, a2, l = traj_def[i]
                        r = QtCore.QRect(x - self.data['l'], y - self.data['l'], 2*self.data['l'], 2*self.data['l'])
                    
                        painter.drawArc(r, a1*16, a2*16)

        painter.end()
            

class Param(QWidget):
    def __init__(self, data, compute_callback, org_callback, poly_cb, obs_cb, scale_cb, path_cb, boxes_cb, coords_cb):
        QWidget.__init__(self)
        self.setWindowTitle("Paramètres")
        #self.resize(200, 300)

        self.data = data
        self.compute_cb = compute_callback
        self.org_cb = org_callback
        self.poly_cb = poly_cb
        self.obstacle_cb = obs_cb
        self.scale_cb = scale_cb
        self.path_cb = path_cb
        self.boxes_cb = boxes_cb
        self.scale = data['scale']
        self.d = data['d']
        self.l = data['l']
        self.s = data['s']
        self.coords = data['coords']

        self.initUI()

    def initUI(self):
        v_lay = QVBoxLayout(self)

        # def origin

        origin_button = QPushButton("Origine", self)
        origin_button.clicked.connect(self.origin_button_clicked)
        v_lay.addWidget(origin_button)

        # coords origin
        h_lay_coords = QHBoxLayout(self)
        N_box_l = QLabel('N/W')
        self.coords_box = QLineEdit()
        self.coords_box.setText(self.coords)
        self.coords_box.textChanged.connect(self.coords_changed)

        #self.l_box.valueChanged.connect(self.boxes_changed)
        
        h_lay_coords.addWidget(N_box_l)
        h_lay_coords.addWidget(self.coords_box)
        v_lay.addLayout(h_lay_coords)

        # digit box d et l

        h_lay_1 = QHBoxLayout(self)

        d_box_l = QLabel('d')

        self.d_box = QDoubleSpinBox()
        self.d_box.setMinimum(0)
        self.d_box.setMaximum(200)
        self.d_box.setSingleStep(0.5)
        self.d_box.setValue(self.d)
        self.d_box.valueChanged.connect(self.boxes_changed)

        l_box_l = QLabel('l')

        self.l_box = QDoubleSpinBox()
        self.l_box.setMinimum(0)
        self.l_box.setMaximum(200)
        self.l_box.setSingleStep(0.5)
        self.l_box.setValue(self.l)
        self.l_box.valueChanged.connect(self.boxes_changed)
        
        s_box_l = QLabel('s')

        self.s_box = QDoubleSpinBox()
        self.s_box.setMinimum(0)
        self.s_box.setMaximum(200)
        self.s_box.setSingleStep(0.5)
        self.s_box.setValue(self.s)
        self.s_box.valueChanged.connect(self.boxes_changed)

        h_lay_1.addWidget(d_box_l)
        h_lay_1.addWidget(self.d_box)
        h_lay_1.addWidget(l_box_l)
        h_lay_1.addWidget(self.l_box)
        h_lay_1.addWidget(s_box_l)
        h_lay_1.addWidget(self.s_box)
        v_lay.addLayout(h_lay_1)

        # Scale

        h_lay_2 = QHBoxLayout(self)
        scale_button = QPushButton("Echelle", self)
        scale_button.clicked.connect(self.scale_button_clicked)
        h_lay_2.addWidget(scale_button)
        v_lay.addLayout(h_lay_2)

        #Poly Drawing

        poly_button = QPushButton("Poly", self)
        poly_button.clicked.connect(self.poly_button_clicked)
        v_lay.addWidget(poly_button)

        # obstacle

        obstacle_button = QPushButton("Add Obstacle", self)
        obstacle_button.clicked.connect(self.obstacle_button_clicked)
        v_lay.addWidget(obstacle_button)

        #path

        path_button = QPushButton("Path", self)
        path_button.clicked.connect(self.path_button_clicked)
        v_lay.addWidget(path_button)

        # compute

        compute_button = QPushButton("Compute", self)
        compute_button.clicked.connect(self.compute_button_clicked)
        v_lay.addWidget(compute_button)


        self.setLayout(v_lay)
        self.setFixedSize(200, 200)

    def compute_button_clicked(self):
        self.compute_cb()

    def origin_button_clicked(self):
        self.org_cb()

    def poly_button_clicked(self):
        self.poly_cb()

    def obstacle_button_clicked(self):
        self.obstacle_cb()

    def scale_button_clicked(self):
        self.scale_cb(self.scale)

    def path_button_clicked(self):
        self.path_cb()

    def boxes_changed(self):
        self.boxes_cb(self.d_box.value(), self.l_box.value(), self.s_box.value())

    def coords_changed(self):
        self.coords_cb(self.coords_box.value())


        
if __name__ == "__main__":
    poly = [(220, 180), (274, 636), (406, 617), (402, 597), (445, 592), (449, 612), (576, 592), (526, 141), (398, 160), (401, 186), (339, 194), (330, 166)]
    data = {"fond":"terrain.png", "type":"obs", "poly":poly, "obstacles":[], "d":10, "l":20, "org":(0, 0), 'scale':0.27, 'coords':''}

    app = QApplication.instance() 
    if not app:
        app = QApplication(sys.argv)
    ter = Terrain()
    param = Param(ter.data, ter.compute, ter.def_origin, ter.drawPoly_cb, ter.drawObstacle_cb, ter.scale_cb, ter.path_cb, ter.digit_boxes_cb, ter.coords_cb)
    window = QWidget()
    layout = QHBoxLayout()
    layout.addWidget(ter)
    layout.addWidget(param)
    window.setLayout(layout)
    window.show()

    app.exec_()
